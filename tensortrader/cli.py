"""Console script for tensortrader."""
import sys

import click

from tensortrader.main import run


# export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system"
# run -->
# conda activate Tensor
# cd tensortrader & python cli.py --ticker BTCUSDT
@click.command()
@click.option("--ticker", help="Ticker code for Binance API")
def main(ticker):
    """Console script for tensortrader."""

    # Ref: https://click.palletsprojects.com/en/8.1.x/
    click.echo("Running Automated Trading Bot")
    run(ticker)


if __name__ == "__main__":
    main()  # pragma: no cover
