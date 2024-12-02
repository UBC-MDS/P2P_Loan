# Import 
import pandas as pd
import click

@click.command()
@click.option('--url', type=str)
def main(url):
    """Download data from url to csv"""
    df = pd.read_csv(url)
    df.to_csv("data/raw/loan_data.csv")

if __name__ == '__main__':
    main()


