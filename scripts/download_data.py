# download_data.py
# authors: Mavis Wong, Yasmin Hassan and Abeba N. Turi
# date: December 05, 2024


import click
import os
import requests
import pandas as pd


def download_csv(url, directory, filename):
    """
    Download a CSV file from the given URL and save it to the specified directory.

    Parameters:
    ----------
    url : str
        The URL of the CSV file to be downloaded.
    directory : str
        The directory where the CSV file will be saved.
    filename : str
        The name to save the file as.

    Returns:
    -------
    None
    """
    # Check if URL is valid
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError('The URL provided is not valid or cannot be accessed.')
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error while accessing the URL: {e}")
    
    # Check if the URL points to a CSV file (simple check by extension)
    if not url.endswith('.csv'):
        raise ValueError('The URL provided does not point to a CSV file.')

    # Ensure directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save the CSV file to the specified directory
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)

    print(f"Data successfully downloaded and saved to {file_path}")

@click.command()
@click.option('--url', type=str, help='URL of dataset to be downloaded')
@click.option('--output_dir', type=str, help='Path to directory where the data will be saved')
@click.option('--filename', type=str, help='Filename to save the downloaded data')
def main(url, output_dir, filename):
    """Downloads CSV data from the web to a local filepath."""
    try:
        download_csv(url, output_dir, filename)
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()
