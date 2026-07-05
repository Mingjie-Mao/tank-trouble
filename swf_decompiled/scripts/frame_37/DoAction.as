_root.loadVariables("includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("gameStarted=3&a=" + Math.random() + "&b=" + Math.random())),"POST");
gotoAndStop("game");
play();
