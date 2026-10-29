@echo off
REM X-Miner - Iron Fish (IRON, FishHash) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: iron-us.kryptex.network:7017!iron.kryptex.network:7017
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=iron-us.kryptex.network:7017!iron.kryptex.network:7017

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   FishHash is limited by random memory reads (4.5 GB dataset): a memory overclock helps, e.g. --gpu-mclock 2700 on the RX 9070 XT (+7%).
set TUNING=

:loop
x-miner.exe --algorithm fishhash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
