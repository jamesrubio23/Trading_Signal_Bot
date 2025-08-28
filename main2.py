import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from ib_async import IB, Forex

from State_Signal_Bot import TradingFunctions, OBTelegramBot, onBarUpdate

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))
CHAT_OB_ID = int(os.getenv("TELEGRAM_CHAT_OB_ID"))


ib = IB()
ib.connect(clientId=1)

contract = Forex("EURUSD")
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='50000 S',
    barSizeSetting='1 min',
    whatToShow='MIDPOINT',
    useRTH=False,
    formatDate=2,
    keepUpToDate=True
)


trading = TradingFunctions(bars)
trading.inicializar_niveles()


bot_wrapper = OBTelegramBot(TOKEN, CHAT_ID, CHAT_OB_ID, ib, trading)
application = bot_wrapper.app_telegram()


on_bar_update = onBarUpdate(trading, bot_wrapper)
bars.updateEvent += on_bar_update


loop = asyncio.get_event_loop()
bot_task = loop.create_task(application.run_polling(drop_pending_updates=False))

try:
    print("Bot iniciado.")
    loop.run_forever()
except KeyboardInterrupt:
    print("Cerrando...")
finally:

    loop.run_until_complete(bot_wrapper.enviar_mensaje("Bot detenido."))

    bars.updateEvent -= on_bar_update
    ib.cancelHistoricalData(bars)
    loop.run_until_complete(application.stop())
    loop.run_until_complete(application.shutdown())
    ib.disconnect()

    ## Para cancelar lo pendiente
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    for t in pending:
        t.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    print("Terminado.")
