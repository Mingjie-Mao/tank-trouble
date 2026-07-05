stop();
player1Joined = false;
onKeyDown = function()
{
   if(Key.getCode() == 77)
   {
      if(!player1Joined)
      {
         if(_root.soundOn)
         {
            _root.soundClick.start();
         }
      }
      var _loc2_ = new Color(player1Controls.background);
      _loc2_.setRGB(16711680);
      player1Controls.fireButton.gotoAndStop(1);
      _loc2_ = new Color(player1Controls.fireButton.background);
      _loc2_.setRGB(16711680);
      player1Controls.activateText.text = "";
      player1Joined = true;
   }
};
Key.addListener(this);
onEnterFrame = function()
{
   if(player1Joined)
   {
      Key.removeListener(this);
      play();
      _root.TANKS = 2;
      _root.AIEnabled = true;
      _root.AIName = "Laika";
      _root.loginInfo.p1n = "Player 1";
      _root.loginInfo.p1e = 0;
      _root.loginInfo.p1bc = 16711680;
      _root.loginInfo.p1tc = 16711680;
      _root.loginInfo.p2n = "Laika";
      _root.loginInfo.p2e = 1000;
      _root.loginInfo.p2bc = 2500134;
      _root.loginInfo.p2tc = 6710886;
      _root.loginInfo.playerNumToControlNum = new Array(1);
      _root.loginInfo.playerNumToControlNum[0] = 1;
      _root.loginInfo.actualRankedPlayers = new Array(2);
      _root.loginInfo.actualRankedPlayers[0] = false;
      _root.loginInfo.actualRankedPlayers[1] = true;
      _root.loginInfo.rankedMatch = true;
      _root.onEnterFrame = undefined;
   }
};
