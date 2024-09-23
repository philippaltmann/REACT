import click

from baselines.random_search import random_search
from evo.eval.evaluation import evo_eval
from evo.eval.fidelity import fidelity
from evo.evo import evo_run
from train.run import run
from train.train_model import train


@click.group()
@click.version_option()
def main():
    """client"""


main.add_command(train)
main.add_command(run)
main.add_command(evo_run)
main.add_command(evo_eval)
main.add_command(random_search)
main.add_command(fidelity)
