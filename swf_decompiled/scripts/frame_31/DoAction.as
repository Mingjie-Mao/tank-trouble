stop();
player1Joined = false;
player2Joined = false;
player3Joined = false;
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
   if(player1Joined && player2Joined && player3Joined)
   {
      Key.removeListener(this);
      onMouseDown = undefined;
      play();
      _root.TANKS = 3;
      _root.loginInfo.p1n = "Player 1";
      _root.loginInfo.p1e = 0;
      _root.loginInfo.p1bc = 16711680;
      _root.loginInfo.p1tc = 16711680;
      _root.loginInfo.p2n = "Player 2";
      _root.loginInfo.p2e = 0;
      _root.loginInfo.p2bc = 65280;
      _root.loginInfo.p2tc = 65280;
      _root.loginInfo.p3n = "Player 3";
      _root.loginInfo.p3e = 0;
      _root.loginInfo.p3bc = 255;
      _root.loginInfo.p3tc = 255;
      _root.loginInfo.playerNumToControlNum = new Array(3);
      _root.loginInfo.playerNumToControlNum[0] = 0;
      _root.loginInfo.playerNumToControlNum[1] = 1;
      _root.loginInfo.playerNumToControlNum[2] = 2;
      _root.loginInfo.actualRankedPlayers = new Array(3);
      _root.loginInfo.actualRankedPlayers[0] = false;
      _root.loginInfo.actualRankedPlayers[1] = false;
      _root.loginInfo.actualRankedPlayers[2] = false;
      _root.onEnterFrame = undefined;
   }
};
onMouseDown = function()
{
   if(!player3Joined && !_root.sound.hitTest(_root._xmouse,_root._ymouse,false) && !_root.settings.hitTest(_root._xmouse,_root._ymouse,false))
   {
      if(_root.soundOn)
      {
         _root.soundClick.start();
      }
      var _loc2_ = new Color(player3Controls.background);
      _loc2_.setRGB(255);
      player3Controls.fireButton.gotoAndStop(1);
      _loc2_ = new Color(player3Controls.fireButton.background);
      _loc2_.setRGB(255);
      player3Controls.activateText.text = "";
      player3Joined = true;
   }
};
