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



    ###################
    ##SWING HIGH/LOW###
    ###################

    def swing_highs_lows(self, swing_length=10):
        """
        Swing Highs and Lows
        A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
        A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

        parameters:
        swing_length: int - the amount of candles to look back and forward to determine the swing high or low

        returns:
        HighLow = 1 if swing high, -1 if swing low
        Level = the level of the swing high or low
        """
        swing_length*=2

        swing_highs_lows = np.where(
            self.df['high']== self.df['high'].shift(-(swing_length//2)).rolling(swing_length).max(), 1,
            np.where(self.df['low']==self.df['low'].shift(-(swing_length//2)).rolling(swing_length).min(), -1, np.nan),
        )

        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                break  

            current = swing_highs_lows[positions[:-1]]
            next = swing_highs_lows[positions[1:]]

            highs = self.df['high'].iloc[positions[:-1]].values
            lows = self.df['low'].iloc[positions[:-1]].values

            next_highs = self.df['high'].iloc[positions[1:]].values
            next_lows = self.df['low'].iloc[positions[1:]].values

            print(f"We have next_highs: {next_highs}")
            print(f"We have next_lows: {next_highs}")

            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            consecutive_lows = (current == -1) & (next == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)


            if not index_to_remove.any():
                break

            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]

        print(f"Positions: {positions}")

        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        self.df["HighLow"] = swing_highs_lows
        self.df["Level"] = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, highs, lows),
            np.nan,
        )







    #################
    ###BoS y Choch###
    #################

    def bos_choch(self, swing_highs_lows, close_break = True):
        """
        BOS - Break of Structure
        CHoCH - Change of Character
        these are both indications of market structure changing

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

        returns:
        BOS = 1 if bullish break of structure, -1 if bearish break of structure
        CHOCH = 1 if bullish change of character, -1 if bearish change of character
        Level = the level of the break of structure or change of character
        BrokenIndex = the index of the candle that broke the level
        """

        swing_highs_lows = swing_highs_lows.copy()

        level_order = []
        highs_lows_order = []

        bos = np.zeros(len(self.df), dtype=np.int32)
        choch = np.zeros(len(self.df), dtype=np.int32)
        level = np.zeros(len(self.df), dtype=np.float32)

        last_positions = []

        for i in range(len(swing_highs_lows["HighLow"])):
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])
                if len(level_order) >= 4:
                    # bullish bos
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-4]
                                < level_order[-2]
                                < level_order[-3]
                                < level_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bearish bos
                    bos[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-4]
                                > level_order[-2]
                                > level_order[-3]
                                > level_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bullish choch
                    choch[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-1]
                                > level_order[-3]
                                > level_order[-4]
                                > level_order[-2]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # bearish choch
                    choch[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-1]
                                < level_order[-3]
                                < level_order[-4]
                                < level_order[-2]
                            )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(self.df), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(self.df), dtype=np.bool_)
            # if the bos is 1 then check if the candles high has gone above the level
            if bos[i] == 1 or choch[i] == 1:
                mask = self.df["close" if close_break else "high"][i + 2 :] > level[i]
            # if the bos is -1 then check if the candles low has gone below the level
            elif bos[i] == -1 or choch[i] == -1:
                mask = self.df["close" if close_break else "low"][i + 2 :] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j
                # if there are any unbroken bos or choch that started before this one and ended after this one then remove them
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        # remove the ones that aren't broken
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        # replace all the 0s with np.nan
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        level = pd.Series(level, name="Level")
        broken = pd.Series(broken, name="BrokenIndex")

        return pd.concat([bos, choch, level, broken], axis=1)


    







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