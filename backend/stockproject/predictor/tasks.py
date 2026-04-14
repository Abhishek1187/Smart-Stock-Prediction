from celery import shared_task
from django.core.management import call_command


@shared_task
def run_market_sync(symbols=None):
    if symbols:
        call_command("sync_market_data", symbols=symbols)
    else:
        call_command("sync_market_data")
