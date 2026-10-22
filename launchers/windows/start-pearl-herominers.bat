@echo off
REM X-Miner - Pearl (PRL) on HeroMiners (prl1... wallet address)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: us.pearl.herominers.com:1200
cd /d "%~dp0"

set WALLET=YOUR_PRL_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us.pearl.herominers.com:1200

:loop
x-miner.exe --algorithm pearlhash --pool %POOL% --wallet %WALLET% --worker %WORKER%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
