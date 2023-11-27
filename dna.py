import torch
import sys

if __name__ == "__main__":
    _, *args = sys.argv

    if len(args) != 1:
        print("USAGE: python dna.py <model number>")
        exit(1)

    model = int(args[0])
    pt_file = torch.load(f"opus_books_weights/tmodel_{model}.pt")
    print(pt_file)
