function createTankIcon(playerNum)
{
   ti = _root.attachMovie("tankIcon","tankIcon" + playerNum,_root.getNextHighestDepth());
   if(playerNum == 1)
   {
      ti.x = 356;
   }
   else
   {
      ti.x = 256;
   }
   ti.y = 315;
   ti._x = ti.x;
   ti._y = ti.y;
   if(playerNum == 1)
   {
      ti._alpha = 100;
   }
   else
   {
      ti._alpha = 0;
   }
   ti.num = playerNum;
   ti.stop();
   new Color(ti.tracks.background).setRGB(_root.loginInfo["p" + playerNum + "bc"]);
   new Color(ti.turretBackground).setRGB(_root.loginInfo["p" + playerNum + "tc"]);
   ti.onEnterFrame = function()
   {
      if(this.num == _root.currentPlayerToSelect)
      {
         if(this.x < 356)
         {
            this.x -= (this.x - 356) * 0.4;
         }
         if(this._alpha < 100)
         {
            this._alpha += 20;
         }
      }
      else if(this.num < _root.currentPlayerToSelect)
      {
         if(this.x < 456)
         {
            this.x -= (this.x - 456) * 0.4;
            if(this._alpha > 0)
            {
               this._alpha -= 20;
            }
         }
         else
         {
            this.removeMovieClip();
         }
      }
      this._y = this.y;
      this._x = this.x;
   };
}
stop();
control1Taken = false;
contro21Taken = false;
contro31Taken = false;
_root.loginInfo.controlNumToPlayerNum = new Array(_root.loginInfo.numUsers);
_root.loginInfo.playerNumToControlNum = new Array(_root.loginInfo.numUsers);
var currentPlayerToSelect = 1;
messageText.text = _root.loginInfo.p1n + ",\nchoose your controls!";
var i = 0;
while(i < _root.loginInfo.numUsers)
{
   createTankIcon(i + 1);
   i++;
}
onEnterFrame = function()
{
   if(currentPlayerToSelect > _root.loginInfo.numUsers)
   {
      Key.removeListener(this);
      onMouseDown = undefined;
      play();
      _root.TANKS = _root.loginInfo.numUsers;
      if(_root.TANKS == 1 && !_root.onlinePlay)
      {
         _root.AIEnabled = true;
         _root.AIName = "Laika";
         _root.TANKS = 2;
         _root.loginInfo.p2n = "Laika";
         _root.loginInfo.p2e = 1000;
         _root.loginInfo.p2bc = 2500134;
         _root.loginInfo.p2tc = 6710886;
      }
      _root.loginInfo.actualRankedPlayers = new Array(3);
      _root.loginInfo.actualRankedPlayers[0] = true;
      _root.loginInfo.actualRankedPlayers[1] = true;
      _root.loginInfo.actualRankedPlayers[2] = true;
      _root.loginInfo.rankedMatch = true;
      _root.onEnterFrame = undefined;
      messageText.text = "";
   }
   else
   {
      messageText.text = _root.loginInfo["p" + currentPlayerToSelect + "n"] + ",\nchoose your controls!";
   }
};
onKeyDown = function()
{
   if(Key.getCode() == 81)
   {
      if(!control1Taken && currentPlayerToSelect <= _root.loginInfo.numUsers)
      {
         if(_root.soundOn)
         {
            _root.soundClick.start();
         }
         control1Taken = true;
         var _loc2_ = new Color(controls1.background);
         _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "bc"]);
         controls1.fireButton.gotoAndStop(1);
         _loc2_ = new Color(controls1.fireButton.background);
         _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "tc"]);
         controls1.playerName.text = _root.loginInfo["p" + currentPlayerToSelect + "n"];
         controls1.activateText.text = "";
         _root.loginInfo.controlNumToPlayerNum[0] = currentPlayerToSelect;
         _root.loginInfo.playerNumToControlNum[currentPlayerToSelect - 1] = 0;
         currentPlayerToSelect++;
      }
   }
   if(Key.getCode() == 77)
   {
      if(!control2Taken && currentPlayerToSelect <= _root.loginInfo.numUsers)
      {
         if(_root.soundOn)
         {
            _root.soundClick.start();
         }
         control2Taken = true;
         _loc2_ = new Color(controls2.background);
         _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "bc"]);
         controls2.fireButton.gotoAndStop(1);
         _loc2_ = new Color(controls2.fireButton.background);
         _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "tc"]);
         controls2.playerName.text = _root.loginInfo["p" + currentPlayerToSelect + "n"];
         controls2.activateText.text = "";
         _root.loginInfo.controlNumToPlayerNum[1] = currentPlayerToSelect;
         _root.loginInfo.playerNumToControlNum[currentPlayerToSelect - 1] = 1;
         currentPlayerToSelect++;
      }
   }
};
Key.addListener(this);
onMouseDown = function()
{
   if(!control3Taken && currentPlayerToSelect <= _root.loginInfo.numUsers && !_root.sound.hitTest(_root._xmouse,_root._ymouse,false) && !_root.settings.hitTest(_root._xmouse,_root._ymouse,false))
   {
      if(_root.soundOn)
      {
         _root.soundClick.start();
      }
      control3Taken = true;
      var _loc2_ = new Color(controls3.background);
      _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "bc"]);
      controls3.fireButton.gotoAndStop(1);
      _loc2_ = new Color(controls3.fireButton.background);
      _loc2_.setRGB(_root.loginInfo["p" + currentPlayerToSelect + "tc"]);
      controls3.playerName.text = _root.loginInfo["p" + currentPlayerToSelect + "n"];
      controls3.activateText.text = "";
      _root.loginInfo.controlNumToPlayerNum[2] = currentPlayerToSelect;
      _root.loginInfo.playerNumToControlNum[currentPlayerToSelect - 1] = 2;
      currentPlayerToSelect++;
   }
};
