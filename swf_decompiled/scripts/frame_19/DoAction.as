stop();
player1Joined = false;
player2Joined = false;
onKeyDown = function()
{
   if(Key.getCode() == 81)
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
   if(Key.getCode() == 77)
   {
      if(!player2Joined)
      {
         if(_root.soundOn)
         {
            _root.soundClick.start();
         }
      }
      _loc2_ = new Color(player2Controls.background);
      _loc2_.setRGB(65280);
      player2Controls.fireButton.gotoAndStop(1);
      _loc2_ = new Color(player2Controls.fireButton.background);
      _loc2_.setRGB(65280);
      player2Controls.activateText.text = "";
      player2Joined = true;
   }
};
Key.addListener(this);
onEnterFrame = function()
{
   if(player1Joined && player2Joined)
   {
      Key.removeListener(this);
      play();
      _root.TANKS = 2;
      _root.loginInfo.p1n = "Player 1";
      _root.loginInfo.p1e = 0;
      _root.loginInfo.p1bc = 16711680;
      _root.loginInfo.p1tc = 16711680;
      _root.loginInfo.p2n = "Player 2";
      _root.loginInfo.p2e = 0;
      _root.loginInfo.p2bc = 65280;
      _root.loginInfo.p2tc = 65280;
      _root.loginInfo.playerNumToControlNum = new Array(2);
      _root.loginInfo.playerNumToControlNum[0] = 0;
      _root.loginInfo.playerNumToControlNum[1] = 1;
      _root.loginInfo.actualRankedPlayers = new Array(2);
      _root.loginInfo.actualRankedPlayers[0] = false;
      _root.loginInfo.actualRankedPlayers[1] = false;
      _root.onEnterFrame = undefined;
   }
};
