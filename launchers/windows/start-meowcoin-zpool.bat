@echo off
REM X-Miner - Meowcoin (MEWC, MeowPow) on zpool (paid in the coin you choose: BTC, LTC, DOGE, ...)
REM Edit WALLET, PAYOUT and WORKER below, then double-click this file. PAYOUT is the coin of WALLET.
REM Pool: meowpow.na.mine.zpool.ca:1327 (zpool lists other regions on its site)
cd /d "%~dp0"

set WALLET=YOUR_PAYOUT_ADDRESS
set PAYOUT=BTC
set WORKER=%COMPUTERNAME%
set POOL=meowpow.na.mine.zpool.ca:1327

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   needs a 12 GB card (the DAG is ~10.6 GB); the memory clock / offset raises it most.
set TUNING=

:loop
x-miner.exe --algorithm meowpow --pool %POOL% --wallet %WALLET% --password c=%PAYOUT% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
