#!/usr/bin/env python3
"""Transcribe local recordings in resumable chunks on Apple silicon Macs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import wave

MODEL = 'mlx-community/whisper-small-mlx'
RATE = 16000


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Audio or video file accessible on this Mac')
    parser.add_argument('--output', required=True, type=Path, help='Dedicated per-recording work directory')
    parser.add_argument('--model', default=MODEL, help='Local model path or MLX model ID')
    parser.add_argument('--chunk-seconds', type=int, default=600)
    parser.add_argument('--overlap-seconds', type=int, default=3)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('This local MLX implementation requires an Apple silicon Mac.')
    if not args.input.is_file():
        parser.error('Input file is missing or inaccessible.')
    if args.chunk_seconds <= 0 or not 0 <= args.overlap_seconds < args.chunk_seconds:
        parser.error('Require chunk-seconds > overlap-seconds >= 0.')
    if not shutil.which('afconvert'):
        parser.error('macOS afconvert was not found.')
    try:
        import numpy as np
        import mlx_whisper
    except Exception as exc:
        parser.error('Local dependencies unavailable (mlx-whisper and numpy): ' + str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    identity = {'source_sha256': digest(args.input), 'model': args.model,
                'chunk_seconds': args.chunk_seconds, 'overlap_seconds': args.overlap_seconds,
                'sample_rate': RATE, 'language': 'ja', 'format_version': 1}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(old.get(k) != v for k, v in identity.items()):
            raise RuntimeError('Input or settings differ from this work directory. Choose a new --output directory.')
    else:
        atomic_json(manifest_path, identity)

    # afconvert writes a local uncompressed working copy; original is never changed.
    with tempfile.TemporaryDirectory(prefix='audio-work-', dir=args.output) as temp:
        wav_path = Path(temp) / 'source.wav'
        subprocess.run(['afconvert', '-f', 'WAVE', '-d', f'LEI16@{RATE}', '-c', '1',
                        str(args.input.resolve()), str(wav_path)], check=True)
        with wave.open(str(wav_path), 'rb') as audio:
            if audio.getframerate() != RATE or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise RuntimeError('Unexpected converted audio format.')
            frames = audio.getnframes()
            if frames <= 0:
                raise RuntimeError('No audio samples found.')
            duration = frames / RATE
            total = math.ceil(duration / args.chunk_seconds)
            print(f'Duration: {duration:.1f}s; chunks: {total}', flush=True)
            for i in range(total):
                path = args.output / f'chunk-{i+1:04d}.json'
                start = i * args.chunk_seconds
                end = min(duration, (i + 1) * args.chunk_seconds)
                if path.exists():
                    chunk = json.loads(path.read_text(encoding='utf-8'))
                    if chunk.get('index') != i + 1 or chunk.get('core_start') != start or chunk.get('core_end') != end or not isinstance(chunk.get('segments'), list):
                        raise RuntimeError(f'Invalid checkpoint: {path}. Inspect before resuming.')
                    print(f'Skipping completed chunk {i+1}/{total}', flush=True)
                    continue
                window_start = max(0, start - args.overlap_seconds)
                window_end = min(duration, end + args.overlap_seconds)
                audio.setpos(round(window_start * RATE))
                pcm = audio.readframes(round((window_end - window_start) * RATE))
                samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
                print(f'Transcribing chunk {i+1}/{total} ({start:.1f}-{end:.1f}s)', flush=True)
                result = mlx_whisper.transcribe(samples, path_or_hf_repo=args.model, language='ja', verbose=False)
                if not isinstance(result, dict) or not isinstance(result.get('segments'), list):
                    raise RuntimeError(f'No timestamped segment result for chunk {i+1}; checkpoint not written.')
                segments = []
                for seg in result['segments']:
                    absolute_start = window_start + float(seg['start'])
                    absolute_end = window_start + float(seg['end'])
                    midpoint = (absolute_start + absolute_end) / 2
                    if start <= midpoint < end and seg.get('text', '').strip():
                        segments.append({'start': round(absolute_start, 2), 'end': round(absolute_end, 2), 'text': seg['text'].strip()})
                atomic_json(path, {'index': i+1, 'core_start': start, 'core_end': end, 'segments': segments})

    all_segments = []
    for i in range(total):
        all_segments.extend(json.loads((args.output / f'chunk-{i+1:04d}.json').read_text(encoding='utf-8'))['segments'])
    output_path = args.output / 'transcript.txt'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=args.output, delete=False) as f:
        temp_output = Path(f.name)
        for seg in all_segments:
            f.write(f"[{int(seg['start']//60):02d}:{int(seg['start']%60):02d}] {seg['text']}\n")
    os.replace(temp_output, output_path)
    atomic_json(manifest_path, dict(identity, duration_seconds=duration, total_chunks=total, complete=True))
    print(f'Complete: {output_path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
