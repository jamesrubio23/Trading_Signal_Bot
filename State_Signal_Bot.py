from ib_async import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import asyncio
from dotenv import load_dotenv

### TELEGRAM ###
from telegram import Bot
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

### INDICADORES

from codigo_indicadores.Indicators import ta_Indicators

###################################################################################################
##################################  TRADING FUNCTIONS  ############################################
###################################################################################################


class TradingFunctions():
    def __init__(self, bars, pip_distance = 0.00005, niveles = {}, niveles_romper={} , df = pd.DataFrame(), indicators = ta_Indicators):
        self.bars = bars
        self.niveles = niveles
        self.niveles_romper = niveles_romper
        self.pip_distance = pip_distance
        self.df = df
        self.indicators = indicators
    
    def bars_to_df(self):
        if self.bars is None or len(self.bars) == 0:
            raise ValueError("Los datos de barras están vacíos.")
        
        self.df = pd.DataFrame(self.bars)[["date", "open", "high", "low", "close"]]
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df.set_index("date", inplace=True)
        return self.df
    
    
    def inicializar_niveles(self):
        
        self.niveles = {
            "Min Asia": [None, "No enviada"],
            "Max Asia": [None, "No enviada"],
            "PDL": [None, "No enviada"],
            "PDH": [None, "No enviada"],
            "PWL": [None, "No enviada"],
            "PWH": [None, "No enviada"],
            "Max 1": [None, "No enviada"],
            "Max 2": [None, "No enviada"],
            "Min 1": [None, "No enviada"],
            "Min 2": [None, "No enviada"]
            }

        self.niveles_maxs = {
            "Max Asia": [self.niveles["Max Asia"][0],"No enviada"],
            "PDH": [self.niveles["PDH"][0],"No enviada"],
            "PWH": [self.niveles["PDH"][0],"No enviada"],
            "Max 1": [self.niveles["PDH"][0],"No enviada"],
            "Max 2": [self.niveles["PDH"][0],"No enviada"]
            }
    
        self.niveles_mins = {
            "Min Asia": [self.niveles["Min Asia"][0],"No enviada"],
            "PDL": [self.niveles["PDL"][0],"No enviada"],
            "PWL": [self.niveles["PWL"][0],"No enviada"],
            "Min 1": [self.niveles["Min 1"][0],"No enviada"],
            "Min 2": [self.niveles["Min 2"][0],"No enviada"]
            }
    
    ##################################
    ### FUNCIONES PARA BOT TELEGRAM###
    ##################################
    
    # Tenemos que crear esas funciones porque queremos que la información este solo en esta clase
    # El bot de telegram solo llamará a las funciones y pedirá a esta clase la información necesaria
    
    def actualizar_nivel(self, nombre: str, valor: float) -> bool:
        if nombre in self.niveles:
            self.niveles[nombre][0] = valor
            self.niveles[nombre][1] = "No enviada"
            return True
        return False
    
    def print_info_comandos(self) -> list[str]:
        if self.df is not None and not self.df.empty:
            return [
                "🔹 **/cambiar nombre_nivel = 1.16 :** Para poner alertas en los niveles que busques",
                "🔹 **/restart:** Para reiniciar a 0 los niveles",
                "🔹 **/status:** Te proporciona el estado actual de los fvgs y el nombre y valor de los niveles"
            ]
        else:
            return ["🔹 No hay dataframe!."]
    

        
    def print_niveles(self) -> str:
        mensaje = "\n **Niveles actuales:**\n"
        for nombre, valor_estado in self.niveles.items():
            valor, estado = valor_estado
            mensaje += f"{nombre}: {valor} ({estado})\n"
        return mensaje
    
    def print_valores_df(self) -> str:
        mensaje = "\n **Valores del df:**\n"
        if self.df is not None and not self.df.empty:
            mensaje += f"Primeros 3 valores:\n{self.df.head(3).to_string()}\n\n"
            mensaje += f"Últimos 5 valores:\n{self.df.tail(5).to_string()}\n"
        return mensaje

    def plot_bars(self):
        path= "grafico.png"
        #path= "C:/Users/jaime/Escritorio/TRADING/IBKR/Bot/bot_prueba_v1/bot/grafico.png"
        if self.df is None or self.df.empty:
            return ""

        plot = util.barplot(self.df, title="EURUSD", upColor="green", downColor="red")
        plt.savefig(path)
        plt.close(plot)

        return path





    ###############
    ###PROXIMIDAD##
    ###############
    def check_proximity(self, pip_distance=0.0005)-> list[str]:
        if self.df is None or self.df.empty:
            return []
        price = self.df["close"].iloc[-1]
        señales = []
        for nombre, nivel in self.niveles.items():
            if nivel[0] is not None and abs(price - nivel[0]) <= pip_distance and nivel[1] == "No enviada":
                mensaje = f"⚠ Señal {nombre}: Precio cerca de {nombre} marcado como {nivel[0]}"
                nivel[1] = "Enviada"
                señales.append(mensaje)
        return señales
    
    #################
    ##SUPERAR NIVEL##
    #################
    def check_break_level(self)-> list[str]:
        if self.df is None or self.df.empty:
            return []
        price = self.df["close"].iloc[-1]

        señal_superar = []
        for nombre, nivel in self.niveles_maxs.items():
            if nivel[0] is not None and price > nivel[0] and nivel[1] == "No enviada":
                mensaje = f"⚠ Señal maximo superada: Precio ha superado {nombre} marcado como {nivel[0]}"
                nivel[1] = "Enviada"
                señal_superar.append(mensaje)
        
        for nombre, nivel in self.niveles_mins.items():
            if nivel[0] is not None and price < nivel[0] and nivel[1] == "No enviada":
                mensaje = f"⚠ Señal Minimo superado: Precio ha superado {nombre} marcado como {nivel[0]}"
                nivel[1] = "Enviada"
                señal_superar.append(mensaje)
        return señal_superar



    ##########
    ###FVGs###
    ##########


    #################
    ###BoS y Choch###
    #################


    def isPivot(self, candle, window):
        """
        function that detects if a candle is a pivot/fractal point
        args: candle index, window before and after candle to test if pivot
        returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default

        Candle = 'Close'
        window=5
        df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)
        """
        if candle-window < 0 or candle+window >= len(self.df):
            return 0
        
        pivotHigh = 1
        pivotLow = 2
        for i in range(candle-window, candle+window+1):
            if self.df.iloc[candle].low > self.df.iloc[i].low:
                pivotLow=0
            if self.df.iloc[candle].high < self.df.iloc[i].high:
                pivotHigh=0
        if (pivotHigh and pivotLow):
            return 3
        elif pivotHigh:
            return pivotHigh
        elif pivotLow:
            return pivotLow
        else:
            return 0

    def pointpos(self, x):
        """
        Function that returns the price position of the pivot point
        df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
        """

        if x['isPivot']==2:
            return x['low']-1e-3
        elif x['isPivot']==1:
            return x['high']+1e-3
        else:
            return np.nan
    
    def detect_structure(self, candle, backcandles, window):
        """
        Attention! window should always be greater than the pivot window! to avoid look ahead bias
        """
        localdf = self.df[candle-backcandles-window:candle-window]  
        highs = localdf[localdf['isPivot'] == 1].high.tail(3).values
        idxhighs = localdf[localdf['isPivot'] == 1].high.tail(3).index
        lows = localdf[localdf['isPivot'] == 2].low.tail(3).values
        idxlows = localdf[localdf['isPivot'] == 2].low.tail(3).index

        pattern_detected = False

        lim1 = 0.005
        lim2 = lim1/3
        if len(highs) == 3 and len(lows) == 3:
            order_condition = (idxlows[0] < idxhighs[0] 
                            < idxlows[1] < idxhighs[1] 
                            < idxlows[2] < idxhighs[2])
            diff_condition = ( 
                                abs(lows[0]-highs[0])>lim1 and 
                                abs(highs[0]-lows[1])>lim2 and
                                abs(highs[1]-lows[1])>lim1 and
                                abs(highs[1]-lows[2])>lim2
                                )
            pattern_1 = (lows[0] < highs[0] and ## Alcista
                lows[1] > lows[0] and lows[1] < highs[0] and
                highs[1] > highs[0] and
                lows[2] > lows[1] and lows[2] < highs[1] and
                highs[2] < highs[1] and highs[2] > lows[2]
                )

            pattern_2 = (lows[0] < highs[0] and
                lows[1] > lows[0] and lows[1] < highs[0] and
                highs[1] > highs[0] and
                lows[2] < lows[1] and
                highs[2] < highs[1] 
                )

            if (order_condition and
                diff_condition and
                (pattern_1 or pattern_2)
            ):
                pattern_detected = True

        if pattern_detected:
            return 1
        else:
            return 0
    

    def apply_indicators(self, window=5):
        """
        Function that applies all the indicators to the dataframe
        """
        self.df['isPivot'] = self.df.apply(lambda x: self.isPivot(x.name, window), axis=1)
        self.df['pointpos'] = self.df.apply(lambda x: self.pointpos(x), axis=1)
        self.df['structure'] = self.df.apply(lambda x: self.detect_structure(x.name, backcandles=100, window=window), axis=1)
        return self.df
    

    def plot_structure(self, backcandles=600, window=5):
        localdf = self.df[-backcandles-window:]
        plt.figure(figsize=(12,6))
        plt.plot(localdf['Close'], label='Close Price', color='blue')
        plt.scatter(localdf.index, localdf['pointpos'], color='red', label='Pivots', marker='o')
        plt.title('Price with Pivot Points')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()


    def detect_bos(self, n1=20, n2=5):
        self.df = self.indicators.EMA()







###################################################################################################
######################################  TELEGRAM  #################################################
###################################################################################################




class OBTelegramBot():
    """
    Class that wraps the Telegram Bot functionality
    """
    def __init__(self, TOKEN, CHAT_ID, CHAT_OB_ID, Ibkr_Connection, Trading_Functions: TradingFunctions):
        self.TOKEN = TOKEN
        self.CHAT_ID = CHAT_ID
        self.CHAT_OB_ID = CHAT_OB_ID
        self.bot = Bot(token=self.TOKEN)
        self.Ibkr_Connection = Ibkr_Connection
        self.Trading_Functions = Trading_Functions
    
    

    async def enviar_mensaje(self, texto):
        await self.bot.send_message(self.CHAT_ID, text=texto)
    
    

    async def enviar_imagen(self, file_path):
        sema = asyncio.Semaphore(1)
        async with sema:
            with open(file_path, "rb") as f:
                await self.bot.send_photo(self.CHAT_ID, photo=f)
    

    async def start(self, update, context):
        await update.message.reply_text("Hola! El bot está en marcha 🚀")
    
    async def echo(self, update, context):
        await update.message.reply_text(update.message.text)

    async def stop(self, update, context):
        await update.message.reply_text("⚠️ Bot detenido. Ya no escucho comandos.")
        await context.application.stop()


    ## Información sobre los comandos
    async def info(self, update, context):
        mensaje = "\n 📊 Puedes hacer lo siguiente con el Bot:**\n\n"

        comandos_info = self.Trading_Functions.print_info_comandos()

        mensaje = "\n 📊 Puedes hacer lo siguiente con el Bot:\n\n" + "\n\n".join(comandos_info)
        await update.message.reply_text(mensaje)


    ###Cambiar
    async def cambiar(self, update, context):
        """
        Comando /cambiar <nombre> = <valor>
        Ejemplo: /cambiar Min Asia = 1.1589
        """
        try:
            await update.message.reply_text(f"Recibido: {update.message.text}")

            texto = update.message.text.replace("/cambiar", "").strip()
            await update.message.reply_text(f"Texto procesado: '{texto}'")
            if "=" not in texto:
                await update.message.reply_text("❌ Formato incorrecto, falta '='")
                return

            nombre, valor = texto.split("=")
            nombre = nombre.strip()
            valor = float(valor.strip())

            await update.message.reply_text(f"Nombre detectado: '{nombre}', Valor detectado: {valor}")


            if self.Trading_Functions.actualizar_nivel(nombre, valor):
                await update.message.reply_text(f"✅ Nivel {nombre} actualizado a {valor}")
            else:
                await update.message.reply_text(f"❌ Nivel {nombre} no existe.")

        except Exception as e:
            await update.message.reply_text(f"⚠ Error: {e}")
    



    ###Reiniciar niveles
    async def restart(self, update, context):
        self.Trading_Functions.inicializar_niveles()
        await update.message.reply_text("🔄 Niveles reiniciados a 0.")

    
    ###Aviso###
    """Falta por arreglar Fvgs"""
    async def status(self, update, context):

        await update.message.reply_text("📊 Solo puedes ver los niveles marcados")
        
        niveles_str = self.Trading_Functions.print_niveles()
        await update.message.reply_text("".join(niveles_str))
    
    ### Para comprobar los valores del df
    async def df_status(self, update, context):
        valores_df_str = self.Trading_Functions.print_valores_df()
        print(valores_df_str)
        #await update.message.reply_text(valores_df_str)

    
    async def enviar_señales(self):
        señales = self.Trading_Functions.check_proximity()
        señal_superar = self.Trading_Functions.check_break_level()
        if señales:
            for mensaje in señales:
                await self.enviar_mensaje(mensaje)
            #plot_path = self.Trading_Functions.plot_bars()
            #await self.enviar_imagen(plot_path)
        if señal_superar:
            for mensaje in señal_superar:
                await self.enviar_mensaje(mensaje)
            #plot_path = self.Trading_Functions.plot_bars()
            #await self.enviar_imagen(plot_path)
        

    def app_telegram(self):
        """Construye y devuelve la Application (handlers enlazados a métodos de instancia)."""

        
        app = Application.builder().token(self.TOKEN).build()
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("cambiar", self.cambiar))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.echo))
        app.add_handler(CommandHandler("stop", self.stop))
        app.add_handler(CommandHandler("restart", self.restart))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("df_status", self.df_status))
        app.add_handler(CommandHandler("info", self.info))


        return app


###################################################################################################
######################################  ON BAR UPDATE  ############################################
###################################################################################################





class onBarUpdate():
    def __init__(self, trading_functions: TradingFunctions, OB_Telegram_Bot: OBTelegramBot):
        self.trading_functions = trading_functions
        self.OB_Telegram_Bot = OB_Telegram_Bot

    def __call__(self, bars, hasNewBar):
        self.trading_functions.bars_to_df()
        asyncio.create_task(self.OB_Telegram_Bot.enviar_señales())