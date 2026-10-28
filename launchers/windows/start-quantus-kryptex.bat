@echo off
REM X-Miner - Quantus (QTC) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: qtc-us.kryptex.network:7049!qtc.kryptex.network:7049
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=qtc-us.kryptex.network:7049!qtc.kryptex.network:7049

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Measured examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%% --gpu-voffset -50   (+4.5%% at the same power)
REM   RTX 3060:   set TUNING=--gpu-mclock 810   (+1%%: memory power goes to the core)
set TUNING=

:loop
x-miner.exe --algorithm quantus --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
