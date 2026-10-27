@echo off
REM X-Miner - Pearl (PRL) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: prl-us.kryptex.network:7048!prl.kryptex.network:7048
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=prl-us.kryptex.network:7048!prl.kryptex.network:7048

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Example:
REM set TUNING=--gpu-plimit 70%% --gpu-cclock 1500 --gpu-mclock 7000 --gpu-coffset 100 --gpu-voffset -50 --gpu-fan 70
set TUNING=

:loop
x-miner.exe --algorithm pearlhash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
