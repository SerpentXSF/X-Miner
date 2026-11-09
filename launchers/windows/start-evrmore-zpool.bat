@echo off
REM X-Miner - Evrmore (EVR, EvrProgPow) on zpool (paid in the coin you choose: BTC, LTC, DOGE, ...)
REM Edit WALLET, PAYOUT and WORKER below, then double-click this file. PAYOUT is the coin of WALLET.
REM Pool: evrprogpow.na.mine.zpool.ca:1330 (zpool lists other regions on its site)
cd /d "%~dp0"

set WALLET=YOUR_PAYOUT_ADDRESS
set PAYOUT=BTC
set WORKER=%COMPUTERNAME%
set POOL=evrprogpow.na.mine.zpool.ca:1330

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   KawPoW-like: the memory clock / offset raises it most; the core clock helps a little.
set TUNING=

:loop
x-miner.exe --algorithm evrprogpow --pool %POOL% --wallet %WALLET% --password c=%PAYOUT% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
