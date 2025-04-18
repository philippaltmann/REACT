import click
from evo.evo import evo_run
from train.run_model import run_model
from train.train_model import train
from plot.plot import plot

@click.group()
@click.version_option()
def main():
    """client"""

main.add_command(train)
main.add_command(run_model)
main.add_command(evo_run)
main.add_command(plot)
