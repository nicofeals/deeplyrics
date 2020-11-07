import codecs
import random

def shuffle_text_paragraphs(file_path, output_file_path):
    """
    Args:
        file_path (string): absolute path of the file whose paragraph must be shuffled.
        output_file_path (string): absolute path of the file to write the paragraphs in.
    Returns:
        No output, write directly the shuffled paragraph in the output file.
    """
    with codecs.open(file_path, mode='r', encoding='utf-8') as f:
        data = f.read()

        # Split on \n\n
        paragraphs = data.split('\n\n')

        # Shuffle splits
        random.shuffle(paragraphs)

    with codecs.open(output_file_path,  mode='w', encoding='utf-8') as output:
        for paragraph in paragraphs:
            output.write(paragraph)

            # Add the line break
            output.write('\n\n')

if __name__ == '__main__':
    shuffle_text_paragraphs('merged_lyrics_metalcore.txt', 'merged_lyrics_metalcore_augmented.txt')