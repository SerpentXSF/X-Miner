@echo off
REM X-Miner - Nexa (NEXA) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: nexa-us.kryptex.network:7026!nexa.kryptex.network:7026
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=nexa-us.kryptex.network:7026!nexa.kryptex.network:7026

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning").
set TUNING=

:loop
x-miner.exe --algorithm nexapow --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
