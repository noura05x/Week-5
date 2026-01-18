import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'commands'))

from commands.eda import eda
from commands.preprocess import preprocess
from commands.embed import embed
from commands.train import train
from commands.pipeline import pipeline
@click.group()
def cli():
    pass

cli.add_command(eda)
cli.add_command(preprocess)
cli.add_command(embed)
cli.add_command(train)
cli.add_command(pipeline)

if __name__ == '__main__':
    cli()