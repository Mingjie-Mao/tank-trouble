_root.loadVariables("includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("gameStarted=" + _root.loginInfo.numUsers + "&x=" + _root.loginInfo.x + "&a=" + Math.random() + "&b=" + Math.random())),"POST");
if(_root.onlinePlay)
{
   gotoAndStop("onlineGame");
   play();
}
else
{
   gotoAndStop("game");
   play();
}
