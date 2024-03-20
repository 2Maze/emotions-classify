import argparse
import os
import pandas as pd
import json

from pydub import AudioSegment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        type=str,
        help='Path to annotations',
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path to output dataset',
    )

    return parser.parse_args()


def clean_label(label: str) -> str:
    return label.lower().split(' ')[0].replace(',', '').replace(' ', '')


def main():
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Parse
    data = pd.read_csv(args.input)
    parse_wav = {}
    for _, row in data.iterrows():
        wav_name = row['№ п/п'].split('_')[0] + '.wav'
        start = sum(int(x) * 60 ** i for i, x in enumerate(reversed(row['Начало'].split(":"))))
        end = sum(int(x) * 60 ** i for i, x in enumerate(reversed(row['Окончание'].split(":"))))
        state = row['Эмоциональное состояние']
        emotion = row['Проявляемая эмоция в рамках эмоционального состояния']

        if wav_name not in parse_wav:
            parse_wav[wav_name] = {}
            parse_wav[wav_name]['state'] = state
            parse_wav[wav_name]['emotions'] = []
        parse_wav[wav_name]['emotions'].append({'start': start, 'end': end, 'emotion': emotion})

    # Create datasets files
    annotations = {}
    for filename in parse_wav.keys():
        try:
            wav_path = os.path.join(os.path.split(args.input)[0], filename)
            wav = AudioSegment.from_wav(wav_path)
            for i, fragment in enumerate(parse_wav[filename]['emotions']):
                segment = wav[fragment['start'] * 1000:fragment['end'] * 1000]
                if segment.duration_seconds == 0.:
                    print(f'Filename {filename} has 0 second of duration!')
                    pass
                else:
                    new_filename = f'{filename.replace(".wav", "")}_{i}.wav'
                    segment.export(os.path.join(args.output, new_filename), format='wav')
                    annotations[new_filename] = {'emotion': clean_label(fragment['emotion']),
                                                 'state': clean_label(parse_wav[filename]['state'])}

        except FileNotFoundError as E:
            print(E)

    # Save annotation file
    with open(os.path.join(args.output, 'annotations.json'), 'w') as outfile:
        json.dump(annotations, outfile, ensure_ascii=False)


if __name__ == '__main__':
    main()
