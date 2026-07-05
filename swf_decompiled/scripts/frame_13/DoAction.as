_root.loadVariables("includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("gameStarted=1&x=" + _root.loginInfo.x + "&a=" + Math.random() + "&b=" + Math.random())),"POST");
gotoAndStop("game");
play();
