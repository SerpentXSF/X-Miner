@echo off
REM X-Miner - Tari (XTM, Cuckaroo29) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: xtm-c29-us.kryptex.network:7040!xtm-c29.kryptex.network:7040
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=xtm-c29-us.kryptex.network:7040!xtm-c29.kryptex.network:7040

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning").
set TUNING=

:loop
x-miner.exe --algorithm cuckaroo29 --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
