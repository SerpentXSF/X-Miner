@echo off
REM X-Miner - Ravencoin (RVN, KawPow) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: rvn-us.kryptex.network:7031!rvn.kryptex.network:7031
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=rvn-us.kryptex.network:7031!rvn.kryptex.network:7031

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Measured examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%% --gpu-mclock 2700   (+7%%; 2800 was unstable on our card)
REM   RTX 3060:   set TUNING=--gpu-moffset 1000 --gpu-cclock 1500   (+7%% at 135 W instead of 169 W)
REM   KawPow is limited by memory bandwidth: memory clock / offset raise it, core clock and voltage do not.
set TUNING=

:loop
x-miner.exe --algorithm kawpow --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
