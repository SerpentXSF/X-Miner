@echo off
REM X-Miner - Beam (BEAM) on 2Miners (your Beam wallet address; TLS)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pool: stratum+ssl://beam.2miners.com:5252
cd /d "%~dp0"

set WALLET=YOUR_BEAM_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=stratum+ssl://beam.2miners.com:5252

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning").
set TUNING=

:loop
x-miner.exe --algorithm beamhash3 --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
