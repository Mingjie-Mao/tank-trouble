function shuffleMessage(m)
{
   var _loc1_ = m.split("&");
   var _loc2_ = 0;
   while(_loc2_ < _loc1_.length)
   {
      var _loc3_ = Math.floor(Math.random() * _loc1_.length);
      var _loc4_ = _loc1_[_loc2_];
      _loc1_[_loc2_] = _loc1_[_loc3_];
      _loc1_[_loc3_] = _loc4_;
      _loc2_ = _loc2_ + 1;
   }
   return _loc1_.join("&");
}
function convertFromHexToRGB(c)
{
   var _loc1_ = {};
   _loc1_.r = c >> 16;
   c ^= _loc1_.r << 16;
   _loc1_.g = c >> 8;
   _loc1_.b = c ^ _loc1_.g << 8;
   _loc1_.a = 100;
   return _loc1_;
}
function decodeMessage(m)
{
   var _loc4_ = {};
   var _loc3_ = Base64.Decode(m).split("&");
   var _loc1_ = 0;
   while(_loc1_ < _loc3_.length)
   {
      var _loc2_ = _loc3_[_loc1_].split("=");
      _loc4_[_loc2_[0]] = _loc2_[1];
      _loc1_ = _loc1_ + 1;
   }
   return _loc4_;
}
function placePlayerButton(x, y, z, players, linkTo, onlinePlay)
{
   var _loc3_ = _root.attachMovie("playerButton",(!onlinePlay ? "p" : "onlineP") + "layerButton" + players,_root.getNextHighestDepth());
   _loc3_.hitZone.tabEnabled = false;
   _loc3_.targetZ = - z;
   _loc3_.transX = x;
   _loc3_.transY = y;
   _loc3_._x = x;
   _loc3_._y = y;
   _loc3_.x = 0;
   _loc3_.y = 0;
   _loc3_.z = - z;
   _loc3_.xScale = 100 * (1 / (- _loc3_.z)) * 100;
   _loc3_.yScale = 100 * (1 / (- _loc3_.z)) * 100;
   _loc3_._xscale = _loc3_.xScale;
   _loc3_._yscale = _loc3_.yScale;
   _loc3_.zSpeed = 0;
   _loc3_.pulsateSpeed = 0.25;
   _loc3_.pulsateCounter = 0;
   _loc3_.linkTo = linkTo;
   _loc3_.pressed = false;
   if(_root.loginInfo.numUsers > 0)
   {
      _loc3_.enabled = players == _root.loginInfo.numUsers;
   }
   if(_loc3_.enabled)
   {
      _loc3_.playerNumber.gotoAndStop(-1 + 2 * players);
      _loc3_.hitZone.onPress = function()
      {
         this._parent.pressed = true;
      };
      _loc3_.hitZone.onReleaseOutside = function()
      {
         this._parent.pressed = false;
      };
      _loc3_.hitZone.onRelease = function()
      {
         _root.onlinePlay = onlinePlay;
         _root.gotoAndPlay(linkTo);
         _root.removePlayerButtons();
      };
   }
   else
   {
      _loc3_.playerNumber.gotoAndStop(2 * players);
      _loc3_.disabledText.gotoAndStop(players >= _root.loginInfo.numUsers ? 1 : 2);
      _loc3_.disabledText._alpha = 100;
   }
   _loc3_.onEnterFrame = function()
   {
      if(this.enabled)
      {
         this.pulsateCounter += this.pulsateSpeed;
         if(!this.hitZone.hitTest(_root._xmouse,_root._ymouse,true))
         {
            this.targetZ = - z + Math.abs(Math.sin(this.pulsateCounter)) * 6;
         }
         else if(this.pressed)
         {
            this.targetZ = - z + 7;
            this.z = - z + 7;
         }
         else
         {
            this.targetZ = - z + 15;
            this.z = - z + 15;
         }
      }
      this.zSpeed += (this.targetZ - this.z) * 4.9;
      this.zSpeed *= 0.1;
      this.z += this.zSpeed;
      this.xScale = 100 * (1 / (- this.z)) * 100;
      this.yScale = 100 * (1 / (- this.z)) * 100;
      this._xscale = this.xScale;
      this._yscale = this.yScale;
      this._x = this.transX + this.x * (100 / (- this.z));
      this._y = this.transY + this.y * (100 / (- this.z));
   };
}
function removePlayerButtons()
{
   _root.playerButton1.removeMovieClip();
   _root.playerButton2.removeMovieClip();
   _root.playerButton3.removeMovieClip();
   _root.onlinePlayerButton1.removeMovieClip();
}
function visitSite()
{
   getUrl("http://www.purup.com", "_blank");
}
function setMousePos(ex, ey)
{
   _root.xMouse = ex;
   _root.yMouse = ey;
}
DEBUG = false;
if(DEBUG)
{
   var debugMessage = _root.createTextField("debugMessage",_root.getNextHighestDepth(),0,0,400,200);
   var debugTextFormat = new TextFormat();
   debugTextFormat.color = 16711680;
   debugTextFormat.size = 64;
   debugTextFormat.font = "Arial";
   debugMessage.setNewTextFormat(debugTextFormat);
   debugMessage.text = "DEBUG!";
}
stop();
soundBullet = new Sound(this);
soundBullet.attachSound("soundBullet");
soundClick = new Sound(this);
soundClick.attachSound("load");
soundExplosion = new Sound(this);
soundExplosion.attachSound("explosion5");
soundExplosion2 = new Sound(this);
soundExplosion2.attachSound("explosion3");
soundExplosion3 = new Sound(this);
soundExplosion3.attachSound("soundFragBomb");
soundBounce0 = new Sound(this);
soundBounce0.attachSound("pingpong");
soundBounce1 = new Sound(this);
soundBounce1.attachSound("pingpong2");
soundLaser = new Sound(this);
soundLaser.attachSound("soundLaser");
soundCrate = new Sound(this);
soundCrate.attachSound("crateSpawn");
soundCrateLand = new Sound(this);
soundCrateLand.attachSound("soundCrateLand");
soundFragment = new Sound(this);
soundFragment.attachSound("soundFragment");
soundFragmentHit = new Sound(this);
soundFragmentHit.attachSound("soundFragmentHit");
soundFragmentHit2 = new Sound(this);
soundFragmentHit2.attachSound("soundFragmentHit2");
soundPoof = new Sound(this);
soundPoof.attachSound("soundPoof");
soundGatlingMotor = new Sound(this);
soundGatlingMotor.attachSound("soundGatlingMotor");
soundGatlingMotorStart = new Sound(this);
soundGatlingMotorStart.attachSound("soundGatlingMotorStart");
soundGatlingMotorStop = new Sound(this);
soundGatlingMotorStop.attachSound("soundGatlingMotorStop");
soundGatlingShot = new Sound(this);
soundGatlingShot.attachSound("soundGatlingShot");
soundDeathRayCharge = new Sound(this);
soundDeathRayCharge.attachSound("soundDeathRayCharge");
soundDeathRayFire = new Sound(this);
soundDeathRayFire.attachSound("soundDeathRayFire");
soundHoming = new Sound(this);
soundHoming.attachSound("soundHoming");
soundHoming2 = new Sound(this);
soundHoming2.attachSound("soundHoming2");
soundHoming3 = new Sound(this);
soundHoming3.attachSound("soundHomingFire");
soundMineLand = new Sound(this);
soundMineLand.attachSound("soundMineLand");
soundMineActivate = new Sound(this);
soundMineActivate.attachSound("soundMineActivate");
soundMineArm = new Sound(this);
soundMineArm.attachSound("soundMineArm");
soundMineDetonate = new Sound(this);
soundMineDetonate.attachSound("soundMineDetonateCharge");
settingsActiveWeapons = new Array("laser","frag","gatling","homing","deathRay");
settingsMaxBullets = 5;
settingsMaxCrates = 3;
settingsCrateSpawnModifier = 1;
settingsPlayRandomMazes = true;
settingsPlayMyCustomMazes = true;
settingsPlayOtherCustomMazes = false;
settingsUseNewMouseControl = false;
Color.prototype.setTint = function(r, g, b, amount)
{
   var _loc2_ = new Object();
   _loc2_.ra = _loc2_.ga = _loc2_.ba = 100 - amount;
   var _loc3_ = amount / 100;
   _loc2_.rb = r * _loc3_;
   _loc2_.gb = g * _loc3_;
   _loc2_.bb = b * _loc3_;
   this.setTransform(_loc2_);
};
String.prototype.trim = function()
{
   var _loc2_ = 0;
   var _loc3_ = this.length - 1;
   while(this.charAt(_loc2_) == " ")
   {
      _loc2_ = _loc2_ + 1;
   }
   while(this.charAt(_loc3_) == " ")
   {
      _loc3_ = _loc3_ - 1;
   }
   return this.substr(_loc2_,_loc3_ - _loc2_ + 1);
};
_root.loginInfo = decodeMessage(initCode);
if(DEBUG)
{
   _root.loginInfo.numUsers = 1;
   _root.loginInfo.p1n = "bbc";
   _root.loginInfo.p1s = 0;
   _root.loginInfo.p1bc = 2500134;
   _root.loginInfo.p1tc = 6710886;
}
if(_root.loginInfo.numUsers > 0)
{
   placePlayerButton(136,340.8000000000001,115,1,"controlsLoggedInPlayers",false);
   placePlayerButton(356,340.8000000000001,115,2,"controlsLoggedInPlayers",false);
   placePlayerButton(576,340.8000000000001,115,3,"controlsLoggedInPlayers",false);
   onKeyDown = function()
   {
      if(Key.getCode() == 48 + parseInt(_root.loginInfo.numUsers) && Key.getCode() >= 49)
      {
         Key.removeListener(this);
         _root.removePlayerButtons();
         _root.gotoAndPlay("controlsLoggedInPlayers");
      }
   };
   Key.addListener(this);
}
else
{
   placePlayerButton(136,340.8000000000001,115,1,"controls1Player",false);
   placePlayerButton(356,340.8000000000001,115,2,"controls2Player",false);
   placePlayerButton(576,340.8000000000001,115,3,"controls3Player",false);
   onKeyDown = function()
   {
      if(Key.getCode() == 49)
      {
         Key.removeListener(this);
         _root.removePlayerButtons();
         _root.gotoAndPlay("controls1Player");
      }
      if(Key.getCode() == 50)
      {
         Key.removeListener(this);
         _root.removePlayerButtons();
         _root.gotoAndPlay("controls2Player");
      }
      if(Key.getCode() == 51)
      {
         Key.removeListener(this);
         _root.removePlayerButtons();
         _root.gotoAndPlay("controls3Player");
      }
   };
   Key.addListener(this);
}
var myMazesFetcher = new MazeDataFetcher(new Array(_root.loginInfo.p1n,_root.loginInfo.p2n,_root.loginInfo.p3n));
var otherPeoplesMazesFetcher = new MazeDataFetcher(new Array());
var myMenu = new ContextMenu();
myMenu.hideBuiltInItems();
var copyrightText = new ContextMenuItem("© 2007-2009 www.purup.com",visitSite);
myMenu.customItems.push(copyrightText);
_root.menu = myMenu;
var bigOlFunc = "function dummyExternalInterfaceFunc() { moveFunc = function(e){var isMSIE = /*@cc_on!@*/false;var offsetX = (window.pageXOffset? window.pageXOffset : (document.body.scrollLeft ? document.body.scrollLeft : document.documentElement.scrollLeft - (isMSIE?2:0)));var offsetY = (window.pageYOffset? window.pageYOffset : (document.body.scrollTop ? document.body.scrollTop : document.documentElement.scrollTop - (isMSIE?2:0)));theTankTroubleGame.setMousePos(e.clientX + offsetX - theTankTroubleGameLeft, e.clientY + offsetY - theTankTroubleGameTop);};" + "resizeFunc = function(){ var curleft = curtop = 0; var obj = theTankTroubleGame; if (obj.offsetParent) { do { curleft += obj.offsetLeft; curtop += obj.offsetTop; } while (obj = obj.offsetParent); } theTankTroubleGameLeft = curleft; theTankTroubleGameTop = curtop; };" + "var theTankTroubleGameLeft = 0; var theTankTroubleGameTop = 0; var stupid = (document.addEventListener == null);" + "var theTankTroubleGame = document.getElementById(\'TankTroubleGame\'); if (theTankTroubleGame == null){ theTankTroubleGame = document.getElementsByName(\'TankTroubleGame\')[0]; }" + "if (!stupid) { document.addEventListener(\'mousemove\', moveFunc, false); window.addEventListener(\'resize\', resizeFunc, false); }else{ document.attachEvent(\'onmousemove\', moveFunc); window.attachEvent(\'onresize\', resizeFunc); }" + "resizeFunc(); return \'OK\'; }()";
if(flash.external.ExternalInterface.available)
{
   flash.external.ExternalInterface.addCallback("setMousePos",null,setMousePos);
   if(String(flash.external.ExternalInterface.call(bigOlFunc)) == "null")
   {
   }
}
onMouseMove = function()
{
   setMousePos(_root._xmouse,_root._ymouse);
};
