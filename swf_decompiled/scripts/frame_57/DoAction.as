function createRoom(num, roomState)
{
   var _loc3_ = _root.attachMovie("roomPanel","room" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
   _loc3_.roomName.text = roomState.firstChild.nodeValue;
   _loc3_.playerCountIcon.stop();
   _loc3_.playerCount.text = roomState.attributes.playercount;
   _loc3_.gameCount.text = "TODO";
   _loc3_.passwordPanel._visible = false;
   if(parseInt(roomState.attributes.passwordprotected) == 0)
   {
      _loc3_.passwordProtected = false;
      _loc3_.padlock._visible = false;
   }
   else
   {
      _loc3_.passwordProtected = true;
      _loc3_.passwordPanel.roomPassword.password = true;
   }
   if(num >= NUMBEROFROOMS - NUMBEROFROOMS % MAXROOMSINAROW)
   {
      _loc3_._x = 356 - (NUMBEROFROOMS - 1) % MAXROOMSINAROW * ROOMPREVIEWWIDTH / 2 + num % MAXROOMSINAROW * ROOMPREVIEWWIDTH;
   }
   else
   {
      _loc3_._x = 356 - (MAXROOMSINAROW - 1) * ROOMPREVIEWWIDTH / 2 + num % MAXROOMSINAROW * ROOMPREVIEWWIDTH;
   }
   _loc3_._y = 100 + Math.floor(num / MAXROOMSINAROW) * ROOMPREVIEWHEIGHT;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.desiredX = _loc3_._x;
   _loc3_.desiredY = _loc3_._y;
   _loc3_.x = _loc3_._x;
   _loc3_.y = _loc3_._y;
   _loc3_.background.useHandCursor = true;
   _loc3_.background.onRelease = function()
   {
   };
   _loc3_.background.tabEnabled = false;
   _loc3_.passwordPanel.onEnterFrame = function()
   {
      if(STATE == 2 && this._visible && Selection.getFocus() == "" + this.roomPassword)
      {
         if(Key.isDown(13))
         {
            if(this.roomPassword.text != "")
            {
               trace("JOIN PROTECTED ROOM: " + this._parent.roomName.text);
               _root.ROOMNAME = this._parent.roomName.text;
               var _loc4_ = new XML();
               var _loc3_ = new XMLNode(1,"joinroom");
               _loc3_.appendChild(new XMLNode(2,this._parent.roomName.text));
               _loc3_.attributes.password = this.roomPassword.text;
               _loc4_.appendChild(_loc3_);
               connection.send(_loc4_);
               STATE = 3;
               removeRooms();
            }
         }
      }
   };
   _loc3_.onMouseUp = function()
   {
      if(this.background.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         if(this.passwordProtected)
         {
            if(!this.passwordPanel._visible)
            {
               this.passwordPanel._visible = true;
               this.background.swapDepths(this.getNextHighestDepth());
               this.passwordPanel.swapDepths(this.background);
               Selection.setFocus(this.passwordPanel.roomPassword);
            }
            else if(STATE == 2 && this.passwordPanel.roomPassword.text != "")
            {
               trace("JOIN PROTECTED ROOM: " + this.roomName.text);
               _root.ROOMNAME = this.roomName.text;
               var _loc3_ = new XML();
               var _loc4_ = new XMLNode(1,"joinroom");
               _loc4_.appendChild(new XMLNode(2,this.roomName.text));
               _loc4_.attributes.password = this.passwordPanel.roomPassword.text;
               _loc3_.appendChild(_loc4_);
               connection.send(_loc3_);
               STATE = 3;
               removeRooms();
            }
         }
         else if(STATE == 2)
         {
            trace("JOIN ROOM: " + this.roomName.text);
            _root.ROOMNAME = this.roomName.text;
            _loc3_ = new XML();
            _loc4_ = new XMLNode(1,"joinroom");
            _loc4_.appendChild(new XMLNode(2,this.roomName.text));
            _loc3_.appendChild(_loc4_);
            connection.send(_loc3_);
            STATE = 3;
            removeRooms();
         }
      }
   };
}
function insertContactingServerIcon()
{
   var _loc3_ = _root.attachMovie("serverInfo","contactingServerIcon",_root.getNextHighestDepth());
   _loc3_._x = Stage.width / 2 - _loc3_._width / 2 + 20;
   _loc3_._y = Stage.height / 2;
   _loc3_._alpha = -30;
   _loc3_.remove = false;
   _loc3_.balls._rotation = 0;
   _loc3_.onEnterFrame = function()
   {
      this.balls._rotation += 4;
      if(this.remove)
      {
         this._alpha -= 10;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      }
      else if(this._alpha < 100)
      {
         this._alpha += 10;
      }
   };
}
function removeContactingServerIcon()
{
   _root.contactingServerIcon.remove = true;
}
function insertChatBox()
{
   var _loc2_ = _root.attachMovie("chatPanel","chatPanel",_root.getNextHighestDepth());
   _loc2_._x = Stage.width / 2;
   _loc2_._y = Stage.height * 19 / 20;
   _loc2_._xscale = 0;
   _loc2_._yscale = 0;
}
function insertMessagesBox()
{
   var _loc2_ = _root.attachMovie("chatMessagesPanel","chatMessagesPanel",_root.getNextHighestDepth());
   _loc2_._x = Stage.width / 2;
   _loc2_._y = 10;
}
function insertCreateRoomBox()
{
   var _loc3_ = _root.attachMovie("createRoomPanel","createRoomPanel",_root.getNextHighestDepth());
   _loc3_._x = Stage.width / 2;
   _loc3_._y = Stage.height * 4 / 5;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.roomName.onSetFocus = function(oldFocus)
   {
      Key.removeListener(_root.settings);
      Key.removeListener(_root.sound);
   };
   _loc3_.roomName.onKillFocus = function(newFocus)
   {
      Key.addListener(_root.settings);
      Key.addListener(_root.sound);
   };
   _loc3_.roomPassword.onSetFocus = function(oldFocus)
   {
      Key.removeListener(_root.settings);
      Key.removeListener(_root.sound);
   };
   _loc3_.roomPassword.onKillFocus = function(newFocus)
   {
      Key.addListener(_root.settings);
      Key.addListener(_root.sound);
   };
   _loc3_.onMouseUp = function()
   {
      if(this.createRoomButton.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         if(STATE == 2 && this.roomName.text != "")
         {
            if(this.roomPassword.text != "")
            {
               trace("JOIN PROTECTED ROOM: " + this.roomName.text);
               _root.ROOMNAME = this.roomName.text;
               var _loc3_ = new XML();
               var _loc4_ = new XMLNode(1,"joinroom");
               _loc4_.appendChild(new XMLNode(2,this.roomName.text));
               _loc4_.attributes.password = this.roomPassword.text;
               _loc3_.appendChild(_loc4_);
               connection.send(_loc3_);
               STATE = 3;
               removeRooms();
            }
            else
            {
               trace("JOIN ROOM: " + this.roomName.text);
               _root.ROOMNAME = this.roomName.text;
               _loc3_ = new XML();
               _loc4_ = new XMLNode(1,"joinroom");
               _loc4_.appendChild(new XMLNode(2,this.roomName.text));
               _loc3_.appendChild(_loc4_);
               connection.send(_loc3_);
               STATE = 3;
               removeRooms();
            }
         }
      }
   };
}
function insertCreateGameBox()
{
   var _loc3_ = _root.attachMovie("createGamePanel","createGamePanel",_root.getNextHighestDepth());
   _loc3_._x = Stage.width / 2;
   _loc3_._y = Stage.height * 4 / 5;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.gameName.onSetFocus = function(oldFocus)
   {
      Key.removeListener(_root.settings);
      Key.removeListener(_root.sound);
   };
   _loc3_.gameName.onKillFocus = function(newFocus)
   {
      Key.addListener(_root.settings);
      Key.addListener(_root.sound);
   };
   _loc3_.onMouseUp = function()
   {
      if(this.createGameButton.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         if(STATE == 5 && this.gameName.text != "")
         {
            trace("JOIN GAME: " + this.gameName.text);
            _root.GAMENAME = this.gameName.text;
            _root.GAMEOWNEDBYME = true;
            var _loc3_ = new XML();
            var _loc4_ = new XMLNode(1,"joingame");
            _loc4_.appendChild(new XMLNode(2,this.gameName.text));
            _loc3_.appendChild(_loc4_);
            connection.send(_loc3_);
            STATE = 6;
            removeGameIcons();
         }
      }
   };
}
function insertLeaveRoomButton()
{
   var _loc3_ = _root.attachMovie("leaveButton","leaveRoomButton",_root.getNextHighestDepth());
   _loc3_._x = Stage.width * 4 / 5;
   _loc3_._y = Stage.height * 4 / 5;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.roomName.text = _root.ROOMNAME;
   _loc3_.onMouseUp = function()
   {
      if(this.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         _root.ROOMNAME = undefined;
         this.targetScale = 0;
         _root.chatPanel.remove = true;
         removeGameIcons();
         removeMemberIcons();
         STATE = 2;
         updateRooms(LATESTSERVERSTATUS);
         insertCreateRoomBox();
      }
   };
}
function updateRooms(doc)
{
   var _loc4_ = doc.firstChild.lastChild.childNodes;
   NUMBEROFROOMS = _loc4_.length;
   var _loc2_ = 0;
   while(_loc2_ < _loc4_.length)
   {
      trace(_loc2_ + ": " + _loc4_[_loc2_].firstChild.nodeValue);
      var _loc3_ = getRoomWithName(_loc4_[_loc2_].firstChild.nodeValue);
      if(_loc3_ != null && _loc3_.targetScale != 0)
      {
         if(_loc2_ >= NUMBEROFROOMS - NUMBEROFROOMS % MAXROOMSINAROW)
         {
            _loc3_.desiredX = 356 - (NUMBEROFROOMS - 1) % MAXROOMSINAROW * ROOMPREVIEWWIDTH / 2 + _loc2_ % MAXROOMSINAROW * ROOMPREVIEWWIDTH;
         }
         else
         {
            _loc3_.desiredX = 356 - (MAXROOMSINAROW - 1) * ROOMPREVIEWWIDTH / 2 + _loc2_ % MAXROOMSINAROW * ROOMPREVIEWWIDTH;
         }
         _loc3_.desiredY = 100 + Math.floor(_loc2_ / MAXROOMSINAROW) * ROOMPREVIEWHEIGHT;
         _loc3_.playerCount.text = _loc4_[_loc2_].attributes.playercount;
         _loc3_.gameCount.text = "TODO";
      }
      else
      {
         createRoom(_loc2_,_loc4_[_loc2_]);
      }
      _loc2_ = _loc2_ + 1;
   }
   for($room in _root)
   {
      if(_root[$room]._name.substr(0,4) == "room")
      {
         if(!checkForName(_root[$room].roomName.text,_loc4_))
         {
            _root[$room].targetScale = 0;
         }
      }
   }
}
function getRoomWithName(name)
{
   for($room in _root)
   {
      if(_root[$room]._name.substr(0,4) == "room")
      {
         if(_root[$room].roomName.text == name)
         {
            return _root[$room];
         }
      }
   }
   return null;
}
function getGameWithName(name)
{
   for($game in _root)
   {
      if(_root[$game]._name.substr(0,4) == "game")
      {
         if(_root[$game].gameName.text == name)
         {
            return _root[$game];
         }
      }
   }
   return null;
}
function getMemberWithName(name)
{
   for($member in _root)
   {
      if(_root[$member]._name.substr(0,6) == "member")
      {
         if(_root[$member].memberName.text == name)
         {
            return _root[$member];
         }
      }
   }
   return null;
}
function checkForName(name, states)
{
   var _loc1_ = 0;
   while(_loc1_ < states.length)
   {
      if(name == states[_loc1_].firstChild.nodeValue)
      {
         return true;
      }
      _loc1_ = _loc1_ + 1;
   }
   return false;
}
function removeRooms()
{
   for($room in _root)
   {
      if(_root[$room]._name.substr(0,4) == "room")
      {
         _root[$room].targetScale = 0;
      }
   }
   _root.createRoomPanel.targetScale = 0;
}
function updateJoinedRoom(doc)
{
   var _loc5_ = doc.firstChild.childNodes[2].childNodes;
   var _loc6_ = doc.firstChild.childNodes[3].childNodes;
   NUMBEROFMEMBERS = _loc5_.length;
   NUMBEROFGAMES = _loc6_.length;
   var _loc2_ = 0;
   while(_loc2_ < _loc6_.length)
   {
      trace(_loc2_ + ": " + _loc6_[_loc2_].firstChild.nodeValue);
      var _loc3_ = getGameWithName(_loc6_[_loc2_].firstChild.nodeValue);
      if(_loc3_ != null && _loc3_.targetScale != 0)
      {
         if(_loc2_ >= NUMBEROFGAMES - NUMBEROFGAMES % MAXGAMESINAROW)
         {
            _loc3_.desiredX = 356 - (NUMBEROFGAMES - 1) % MAXGAMESINAROW * GAMEPREVIEWWIDTH / 2 + _loc2_ % MAXGAMESINAROW * GAMEPREVIEWWIDTH;
         }
         else
         {
            _loc3_.desiredX = 356 - (MAXGAMESINAROW - 1) * GAMEPREVIEWWIDTH / 2 + _loc2_ % MAXGAMESINAROW * GAMEPREVIEWWIDTH;
         }
         _loc3_.desiredY = 100 + Math.floor(_loc2_ / MAXGAMESINAROW) * GAMEPREVIEWHEIGHT;
      }
      else
      {
         createGameIcon(_loc2_,_loc6_[_loc2_]);
      }
      _loc2_ = _loc2_ + 1;
   }
   for($game in _root)
   {
      if(_root[$game]._name.substr(0,4) == "game")
      {
         if(!checkForName(_root[$game].gameName,_loc6_))
         {
            _root[$game].removed = true;
         }
      }
   }
   _loc2_ = 0;
   while(_loc2_ < _loc5_.length)
   {
      trace(_loc2_ + ": " + _loc5_[_loc2_].firstChild.nodeValue);
      var _loc4_ = getMemberWithName(_loc5_[_loc2_].firstChild.nodeValue);
      if(_loc4_ != null)
      {
         if(_loc2_ >= NUMBEROFMEMBERS - NUMBEROFMEMBERS % MAXMEMBERSINAROW)
         {
            _loc4_.desiredX = 70 - (NUMBEROFMEMBERS - 1) % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH / 2 + _loc2_ % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH;
         }
         else
         {
            _loc4_.desiredX = 70 - (MAXMEMBERSINAROW - 1) * MEMBERPREVIEWWIDTH / 2 + _loc2_ % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH;
         }
         _loc4_.desiredY = 100 + Math.floor(_loc2_ / MAXMEMBERSINAROW) * MEMBERPREVIEWHEIGHT;
      }
      else
      {
         createMemberIcon(_loc2_,_loc5_[_loc2_]);
      }
      _loc2_ = _loc2_ + 1;
   }
   for($member in _root)
   {
      if(_root[$member]._name.substr(0,6) == "member")
      {
         if(!checkForName(_root[$member].memberName.text,_loc5_))
         {
            _root[$member].targetScale = 0;
         }
      }
   }
}
function createGameIcon(num, gameState)
{
   var _loc3_ = _root.attachMovie("gamePanel","game" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
   _loc3_.gameName.text = gameState.firstChild.nodeValue;
   if(num >= NUMBEROFGAMES - NUMBEROFGAMES % MAXGAMESINAROW)
   {
      _loc3_._x = 356 - (NUMBEROFGAMES - 1) % MAXGAMESINAROW * GAMEPREVIEWWIDTH / 2 + num % MAXGAMESINAROW * GAMEPREVIEWWIDTH;
   }
   else
   {
      _loc3_._x = 356 - (MAXGAMESINAROW - 1) * GAMEPREVIEWWIDTH / 2 + num % MAXGAMESINAROW * GAMEPREVIEWWIDTH;
   }
   _loc3_._y = 100 + Math.floor(num / MAXGAMESINAROW) * GAMEPREVIEWHEIGHT;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.desiredX = _loc3_._x;
   _loc3_.desiredY = _loc3_._y;
   _loc3_.x = _loc3_._x;
   _loc3_.y = _loc3_._y;
   _loc3_.background.useHandCursor = true;
   _loc3_.background.onRelease = function()
   {
   };
   _loc3_.background.tabEnabled = false;
   _loc3_.onMouseUp = function()
   {
      if(this.background.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         trace("JOIN GAME: " + this.gameName.text);
         _root.GAMENAME = this.gameName.text;
         _root.GAMEOWNEDBYME = false;
         var _loc3_ = new XML();
         var _loc4_ = new XMLNode(1,"joingame");
         _loc4_.appendChild(new XMLNode(2,this.gameName.text));
         _loc3_.appendChild(_loc4_);
         connection.send(_loc3_);
         STATE = 6;
         removeGameIcons();
      }
   };
}
function createMemberIcon(num, memberState)
{
   var _loc3_ = _root.attachMovie("memberPanel","member" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
   _loc3_.memberName.text = memberState.firstChild.nodeValue;
   _loc3_.tankIcon.stop();
   if(num >= NUMBEROFMEMBERS - NUMBEROFMEMBERS % MAXMEMBERSINAROW)
   {
      _loc3_._x = 70 - (NUMBEROFMEMBERS - 1) % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH / 2 + num % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH;
   }
   else
   {
      _loc3_._x = 70 - (MAXMEMBERSINAROW - 1) * MEMBERPREVIEWWIDTH / 2 + num % MAXMEMBERSINAROW * MEMBERPREVIEWWIDTH;
   }
   _loc3_._y = 100 + Math.floor(num / MAXMEMBERSINAROW) * MEMBERPREVIEWHEIGHT;
   _loc3_._xscale = 0;
   _loc3_._yscale = 0;
   _loc3_.desiredX = _loc3_._x;
   _loc3_.desiredY = _loc3_._y;
   _loc3_.x = _loc3_._x;
   _loc3_.y = _loc3_._y;
   _loc3_.tankIcon.useHandCursor = true;
   _loc3_.tankIcon.onRelease = function()
   {
   };
   _loc3_.tankIcon.tabEnabled = false;
   _loc3_.onMouseUp = function()
   {
      if(this.tankIcon.hitTest(_root._xmouse,_root._ymouse,true) && _root.AUTHENTICATED)
      {
         if(_root.loginInfo.p1n != this.memberName.text)
         {
            _root.chatPanel.activatePrivateChat(this.memberName.text);
         }
      }
   };
}
function removeGameIcons()
{
   for($game in _root)
   {
      if(_root[$game]._name.substr(0,4) == "game")
      {
         _root[$game].targetScale = 0;
      }
   }
   _root.createGamePanel.targetScale = 0;
}
function removeMemberIcons()
{
   for($member in _root)
   {
      if(_root[$member]._name.substr(0,6) == "member")
      {
         _root[$member].targetScale = 0;
      }
   }
}
function setupMaze(doc)
{
   var _loc15_ = doc.firstChild;
   var _loc14_ = _loc15_.attributes.width;
   var _loc11_ = _loc15_.attributes.height;
   var _loc7_ = new Array(_loc14_);
   var _loc10_ = 0;
   while(_loc10_ < _loc14_)
   {
      _loc7_[_loc10_] = new Array(_loc11_);
      var _loc5_ = 0;
      while(_loc5_ < _loc11_)
      {
         _loc7_[_loc10_][_loc5_] = new Array(0,0,0);
         _loc5_ = _loc5_ + 1;
      }
      _loc10_ = _loc10_ + 1;
   }
   var _loc3_ = 0;
   while(_loc3_ < _loc11_)
   {
      var _loc2_ = 0;
      while(_loc2_ < _loc14_)
      {
         var _loc4_ = _loc15_.childNodes[_loc2_].childNodes[_loc3_];
         trace(_loc4_);
         var _loc6_ = false;
         var _loc8_ = false;
         var _loc9_ = false;
         if(_loc4_.attributes.leftwall == 1)
         {
            _loc8_ = true;
         }
         if(_loc4_.attributes.topwall == 1)
         {
            _loc9_ = true;
         }
         if(_loc4_.attributes.tile == 1)
         {
            _loc6_ = true;
         }
         if(_loc6_)
         {
            _loc7_[_loc2_][_loc3_][0] = 1;
         }
         if(_loc9_)
         {
            _loc7_[_loc2_][_loc3_][1] = 1;
         }
         if(_loc8_)
         {
            _loc7_[_loc2_][_loc3_][2] = 1;
         }
         if(_loc6_)
         {
            grounds.push({x:_loc2_,y:_loc3_});
         }
         _loc2_ = _loc2_ + 1;
      }
      _loc3_ = _loc3_ + 1;
   }
   WIDTH = _loc7_.length;
   HEIGHT = _loc7_[0].length;
   SCALE = Math.min((MOVIEHEIGHT - HEIGHTTOBOTTOM) / (HEIGHT + 0.125),MOVIEWIDTH / (WIDTH + 0.125));
   trace(WIDTH + ", " + HEIGHT + ": " + SCALE);
   _root.createEmptyMovieClip("game",0);
   drawMaze(_loc7_,SCALE);
}
function setupGame(doc)
{
   _root.GAMEID = doc.firstChild.attributes.id;
   var _loc3_ = doc.firstChild.childNodes;
   trace(_loc3_.length);
   var _loc2_ = 0;
   while(_loc2_ < _loc3_.length)
   {
      trace(_loc2_ + ": " + _loc3_[_loc2_].attributes.id);
      setupPlayer(_loc3_[_loc2_].attributes.id,parseFloat(_loc3_[_loc2_].childNodes[2].firstChild.nodeValue),parseFloat(_loc3_[_loc2_].childNodes[3].firstChild.nodeValue),parseInt(_loc3_[_loc2_].childNodes[1].firstChild.nodeValue),_root.SCALE);
      _loc2_ = _loc2_ + 1;
   }
}
function updateGame(doc)
{
   var _loc3_ = doc.firstChild.childNodes;
   trace(_loc3_.length);
   var _loc2_ = 0;
   while(_loc2_ < _loc3_.length)
   {
      if(_loc3_[_loc2_].attributes.id != _root.MYID)
      {
         trace(_loc2_ + ": " + _loc3_[_loc2_].attributes.id);
         updatePlayer(_loc3_[_loc2_].attributes.id,parseFloat(_loc3_[_loc2_].childNodes[2].firstChild.nodeValue),parseFloat(_loc3_[_loc2_].childNodes[3].firstChild.nodeValue),parseInt(_loc3_[_loc2_].childNodes[1].firstChild.nodeValue),parseFloat(_loc3_[_loc2_].childNodes[4].firstChild.nodeValue),parseFloat(_loc3_[_loc2_].childNodes[5].firstChild.nodeValue));
      }
      _loc2_ = _loc2_ + 1;
   }
}
function requestAuthenticatedPlayerInfo()
{
   var _loc6_ = new LoadVars();
   _loc6_.onLoad = function(success)
   {
      if(success)
      {
         this.response = _root.decodeMessage(this.r);
         if(this.response.password != undefined)
         {
            var _loc5_ = new XML();
            var _loc3_ = new XMLNode(1,"login");
            _loc5_.appendChild(_loc3_);
            var _loc4_ = new XMLNode(1,"username");
            var _loc6_ = new XMLNode(1,"password");
            _loc3_.appendChild(_loc4_);
            _loc3_.appendChild(_loc6_);
            _loc4_.appendChild(new XMLNode(3,_root.loginInfo.p1n));
            _loc6_.appendChild(new XMLNode(3,unescape(this.response.password)));
            connection.send(_loc5_);
         }
         else
         {
            displayErrorMessage("Server error. Please log out and back in again.");
         }
      }
      else
      {
         displayErrorMessage("Server error. Please wait a while and restart the game.");
      }
   };
   _loc6_.load("http://www.tanktrouble.com/includes/getUserAuthentication.php?q=" + Base64.Encode(shuffleMessage("username=" + _root.loginInfo.p1n + "&a=" + Math.random() + "&b=" + Math.random())));
}
function checkJoinResult(status, message)
{
   if(status != 1)
   {
      displayErrorMessage(message);
   }
   return status == 1;
}
function setupPlayer(id, startX, startY, angle, scale)
{
   var _loc2_ = _root.game.attachMovie("tank","tank" + id,_root.game.getNextHighestDepth());
   _loc2_._x = (startX + 0.5) * scale + _root.game.mazemc._x;
   _loc2_._y = (startY + 0.5) * scale + _root.game.mazemc._y;
   _loc2_._rotation = angle;
   _loc2_._xscale = 0.55 * scale;
   _loc2_._yscale = 0.55 * scale;
   _loc2_.base.gotoAndStop(1);
   _loc2_.turret.gotoAndStop(1);
   var _loc4_ = -1;
   var _loc8_ = parseInt(PLAYERS[id].bc);
   var _loc7_ = parseInt(PLAYERS[id].tc);
   _loc2_.baseColor = convertFromHexToRGB(_loc8_);
   _loc2_.turretColor = convertFromHexToRGB(_loc7_);
   _loc2_.username = PLAYERS[id].n;
   if(id == _root.MYID)
   {
      _loc4_ = _root.loginInfo.playerNumToControlNum[0];
      _root.MYTANK = _loc2_;
   }
   if(PLAYERS[id] == undefined)
   {
      PLAYERS[id] = new Object();
   }
   PLAYERS[id].tank = _loc2_;
   PLAYERS[id].speed = 0;
   PLAYERS[id].dAngle = 0;
   switch(_loc4_)
   {
      case 0:
         _loc2_.KEYTURNLEFT = 83;
         _loc2_.KEYFORWARD = 69;
         _loc2_.KEYTURNRIGHT = 70;
         _loc2_.KEYBACKUP = 68;
         _loc2_.KEYFIRE = 81;
         _loc2_.mouseTank = false;
         break;
      case 1:
         _loc2_.KEYTURNLEFT = 37;
         _loc2_.KEYFORWARD = 38;
         _loc2_.KEYTURNRIGHT = 39;
         _loc2_.KEYBACKUP = 40;
         _loc2_.KEYFIRE = 77;
         _loc2_.mouseTank = false;
         break;
      case 2:
         _loc2_.mouseTank = true;
         Mouse.hide();
         _root.attachMovie("scopeCross","scopeCross",_root.getNextHighestDepth());
         _root.attachMovie("scopeCircle","scopeCircle",_root.getNextHighestDepth());
         deltaX = _root.game.mazemc._xmouse - _loc2_._x;
         deltaY = _root.game.mazemc._ymouse - _loc2_._y;
         deltaLength = Math.sqrt(Math.pow(deltaX,2) + Math.pow(deltaY,2));
         _root.scopeCross._x = _root._xmouse;
         _root.scopeCross._y = _root._ymouse;
         if(deltaLength > 60)
         {
            _root.scopeCircle._x = _root.game._x + _loc2_._x + deltaX / deltaLength * 60;
            _root.scopeCircle._y = _root.game._y + _loc2_._y + deltaY / deltaLength * 60;
         }
         else
         {
            _root.scopeCircle._x = _root._xmouse;
            _root.scopeCircle._y = _root._ymouse;
         }
   }
   var _loc6_ = new Color(_loc2_.base.background);
   _loc6_.setTint(_loc2_.baseColor.r,_loc2_.baseColor.g,_loc2_.baseColor.b,_loc2_.baseColor.a);
   _root.setEquipment(_loc2_,"none");
   _root.setWeapon(_loc2_,STARTWEAPON);
}
function updatePlayer(id, newX, newY, angle, newSpeed, newDAngle)
{
   PLAYERS[id].tank.x = newX;
   PLAYERS[id].tank.y = newY;
   PLAYERS[id].tank._rotation = angle;
   PLAYERS[id].speed = newSpeed;
   PLAYERS[id].dAngle = newDAngle;
}
function updatePlayerMetaInfo(doc)
{
   var _loc1_ = doc.firstChild.attributes.id;
   if(PLAYERS[_loc1_] == undefined)
   {
      PLAYERS[_loc1_] = new Object();
   }
   PLAYERS[_loc1_].bc = doc.firstChild.firstChild.attributes.base;
   PLAYERS[_loc1_].tc = doc.firstChild.firstChild.attributes.turret;
}
function displayErrorMessage(message)
{
   _root.chatMessagesPanel.insertMessage(message,"room","Error");
   trace("DISPLAYING ERROR MESSAGE: " + message);
}
function displayCountdown(count)
{
   trace("DISPLAYING COUNTDOWN: " + count);
}
function drawMaze(maze, scale)
{
   _root.game.createEmptyMovieClip("mazebg",-1000);
   _root.game.createEmptyMovieClip("mazemc",_root.game.getNextHighestDepth());
   var mazeWidth = Math.floor(maze.length * scale);
   var mazeHeight = Math.floor(maze[0].length * scale);
   var lineThickness = Math.floor(scale / 16);
   var edgeThickness = 1;
   with(_root.game.mazebg)
   {
      lineStyle(undefined,0,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            if(maze[x][y][0] != 0)
            {
               moveTo(Math.floor(x * scale) - lineThickness - edgeThickness,Math.floor(y * scale) - lineThickness - edgeThickness);
               beginFill(0,100);
               lineTo(Math.floor((x + 1) * scale) + lineThickness + edgeThickness,Math.floor(y * scale) - lineThickness - edgeThickness);
               lineTo(Math.floor((x + 1) * scale) + lineThickness + edgeThickness,Math.floor((y + 1) * scale) + lineThickness + edgeThickness);
               lineTo(Math.floor(x * scale) - lineThickness - edgeThickness,Math.floor((y + 1) * scale) + lineThickness + edgeThickness);
               endFill();
            }
            y++;
         }
         x++;
      }
      lineStyle(undefined,0,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            if(maze[x][y][0] != 0)
            {
               moveTo(Math.floor(x * scale),Math.floor(y * scale));
               beginFill(15132390,100);
               lineTo(Math.floor((x + 1) * scale),Math.floor(y * scale));
               lineTo(Math.floor((x + 1) * scale),Math.floor((y + 1) * scale));
               lineTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
               endFill();
            }
            y++;
         }
         x++;
      }
   }
   with(_root.game.mazemc)
   {
      lineStyle(2 * lineThickness,5066061,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            moveTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
            if(maze[x][y][1] != 0)
            {
               lineTo(Math.floor((x + 1) * scale),Math.floor((y + 1) * scale));
            }
            moveTo(Math.floor(x * scale),Math.floor(y * scale));
            if(maze[x][y][2] != 0)
            {
               lineTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
            }
            y++;
         }
         x++;
      }
      var x = 0;
      while(x < maze.length)
      {
         if(maze[x][0][0] != 0)
         {
            moveTo(Math.floor(x * scale),0);
            lineTo(Math.floor((x + 1) * scale),0);
         }
         if(maze[x][maze[x].length - 1][0] != 0)
         {
            moveTo(Math.floor(x * scale),Math.floor(maze[x].length * scale));
            lineTo(Math.floor((x + 1) * scale),Math.floor(maze[x].length * scale));
         }
         x++;
      }
      var y = 0;
      while(y < maze[0].length)
      {
         if(maze[0][y][0] != 0)
         {
            moveTo(0,Math.floor((y + 1) * scale));
            lineTo(0,Math.floor(y * scale));
         }
         if(maze[maze.length - 1][y][0] != 0)
         {
            moveTo(Math.floor(maze.length * scale),Math.floor((y + 1) * scale));
            lineTo(Math.floor(maze.length * scale),Math.floor(y * scale));
         }
         y++;
      }
   }
}
function setWeapon(owner, weapon)
{
   owner.currentWeapon = weapon;
   var _loc0_ = null;
   if((_loc0_ = weapon) === "bullet")
   {
      owner.turret.gotoAndStop(1);
      owner.hitPointsFront = new Array();
      owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
      owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
      owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
      owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
      owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 16 * 11};
      owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 16 * 11};
   }
   var _loc2_ = new Color(owner.turret.background);
   _loc2_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
}
function setEquipment(owner, equ)
{
   owner.equipment.removeMovieClip();
   owner.currentEquipment = equ;
   var _loc0_ = null;
   if((_loc0_ = equ) !== "none")
   {
   }
}
function fireWeapon(owner, weapon)
{
   switch(weapon)
   {
      case "bullet":
         fireBullet(owner);
         break;
      case "laser":
         fireLaser(owner);
         break;
      case "frag":
         fireFrag(owner);
         break;
      case "gatling":
         fireGatling(owner);
         break;
      case "deathRay":
         fireDeathRay(owner);
         break;
      case "homing":
         fireHoming(owner);
         break;
      case "mine":
         layMine(owner);
         break;
      case "remote":
         fireRemote(owner);
         break;
      case "electric":
         fireElectric(owner);
   }
}
function fireBullet(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundBullet.start();
   }
   bulletDepth = _root.game.getNextHighestDepth();
   bulletName = "bullet" + bulletDepth;
   bullet = _root.game.attachMovie("bullet",bulletName,bulletDepth);
   owner.swapDepths(bullet);
   bullet.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet._x = bullet.x;
   bullet._y = bullet.y;
   bullet._xscale = 100 * (_root.SCALE / 50);
   bullet._yscale = 100 * (_root.SCALE / 50);
   bullet.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.lifetime = BULLETLIFETIME;
   bullet.deadly = BULLETDEADLY;
   bullet.owner = owner;
   owner.bulletsFired = owner.bulletsFired + 1;
}
stop();
var HOST = "tyrael.fork.dk";
var PORT = 10101;
var MYID;
var ROOMNAME;
var NUMBEROFROOMS;
var MAXROOMSINAROW = 4;
var ROOMPREVIEWWIDTH = 135;
var ROOMPREVIEWHEIGHT = 60;
var GAMENAME;
var GAMEOWNEDBYME = false;
var NUMBEROFGAMES;
var MAXGAMESINAROW = 4;
var GAMEPREVIEWWIDTH = 60;
var GAMEPREVIEWHEIGHT = 60;
var NUMBEROFMEMBERS;
var MAXMEMBERSINAROW = 2;
var MEMBERPREVIEWWIDTH = 60;
var MEMBERPREVIEWHEIGHT = 60;
var LATESTSERVERSTATUS;
var LATESTROOMSTATUS;
var AUTHENTICATED = false;
var PLAYERS = new Object();
var MYTANK;
var MODE = "Death Match";
var STATE = 0;
MOVIEWIDTH = 692;
MOVIEHEIGHT = 480;
HEIGHTTOBOTTOM = 80;
STARTWEAPON = "bullet";
BULLETSPEED = 4.5;
BULLETLIFETIME = 250;
BULLETHITCHECKINTERVALS = 7;
BULLETDEADLY = 0;
NUMBEROFFRAGMENTS = 8;
NUMBEROFSMOKECLOUDS = 10;
NUMBEROFPUFFCLOUDS = 0;
NUMBEROFDUSTCLOUDS = 20;
MAXSHAKE = 8;
shake = 0;
gameX = gameY = 0;
var oldTurnLeft = false;
var oldTurnRight = false;
var oldForward = false;
var oldBackup = false;
var oldFire = false;
var updateCount = 0;
onEnterFrame = function()
{
   if(STATE == 8)
   {
      updateCount++;
      if(Key.isDown(MYTANK.KEYTURNLEFT))
      {
         turnLeft = true;
      }
      else
      {
         turnLeft = false;
      }
      if(Key.isDown(MYTANK.KEYFORWARD))
      {
         forward = true;
      }
      else
      {
         forward = false;
      }
      if(Key.isDown(MYTANK.KEYTURNRIGHT))
      {
         turnRight = true;
      }
      else
      {
         turnRight = false;
      }
      if(Key.isDown(MYTANK.KEYBACKUP))
      {
         backup = true;
      }
      else
      {
         backup = false;
      }
      if(Key.isDown(MYTANK.KEYFIRE))
      {
         fire = true;
      }
      else
      {
         fire = false;
      }
      if(oldTurnLeft != turnLeft || oldTurnRight != turnRight || oldForward != forward || oldBackup != backup || oldFire != fire || updateCount >= 2)
      {
         updateCount = 0;
         trace("SEND PLAYER STATE");
         var _loc7_ = new XML();
         var _loc2_ = new XMLNode(1,"playerstate");
         _loc7_.appendChild(_loc2_);
         _loc2_.attributes.id = _root.MYID;
         var _loc12_ = new XMLNode(1,"alive");
         _loc2_.appendChild(_loc12_);
         _loc12_.appendChild(new XMLNode(3,"1"));
         var _loc6_ = new XMLNode(1,"direction");
         var _loc9_ = new XMLNode(1,"x");
         var _loc8_ = new XMLNode(1,"y");
         _loc2_.appendChild(_loc6_);
         _loc2_.appendChild(_loc9_);
         _loc2_.appendChild(_loc8_);
         _loc6_.appendChild(new XMLNode(3,_root.MYTANK._rotation));
         _loc9_.appendChild(new XMLNode(3,_root.MYTANK._x - _root.game.mazemc._x + ""));
         _loc8_.appendChild(new XMLNode(3,_root.MYTANK._y - _root.game.mazemc._y + ""));
         var _loc10_ = new XMLNode(1,"speed");
         var _loc11_ = new XMLNode(1,"deltadirection");
         _loc2_.appendChild(_loc10_);
         _loc2_.appendChild(_loc11_);
         var _loc4_ = 0;
         if(forward)
         {
            _loc4_ += PLAYERS[_root.MYID].tank.forwardSpeed;
         }
         if(backup)
         {
            _loc4_ -= PLAYERS[_root.MYID].tank.backUpSpeed;
         }
         _loc10_.appendChild(new XMLNode(3,_loc4_ + ""));
         var _loc5_ = 0;
         if(turnLeft)
         {
            _loc5_ -= PLAYERS[_root.MYID].tank.turnSpeed;
         }
         if(turnRight)
         {
            _loc5_ += PLAYERS[_root.MYID].tank.turnSpeed;
         }
         _loc11_.appendChild(new XMLNode(3,_loc5_ + ""));
         var _loc13_ = new XMLNode(1,"shots");
         _loc2_.appendChild(_loc13_);
         connection.send(_loc7_);
      }
      oldTurnLeft = turnLeft;
      oldTurnRight = turnRight;
      oldForward = forward;
      oldBackup = backup;
      oldFire = fire;
      for(var _loc3_ in PLAYERS)
      {
         trace(_loc3_);
         if(_loc3_ != _root.MYID)
         {
            PLAYERS[_loc3_].tank._rotation += PLAYERS[_loc3_].dAngle;
            PLAYERS[_loc3_].tank.x += Math.cos((PLAYERS[_loc3_].tank._rotation - 90) * 3.141592653589793 / 180) * PLAYERS[_loc3_].speed;
            PLAYERS[_loc3_].tank.y += Math.sin((PLAYERS[_loc3_].tank._rotation - 90) * 3.141592653589793 / 180) * PLAYERS[_loc3_].speed;
         }
      }
   }
};
insertMessagesBox();
insertContactingServerIcon();
var connection = new XMLSocket();
connection.onConnect = function(success)
{
   removeContactingServerIcon();
   if(success)
   {
      STATE = 1;
      requestAuthenticatedPlayerInfo();
   }
   else
   {
      displayErrorMessage("Couldn\'t connect to game server. Please wait a while and restart the game.");
   }
};
connection.onXML = function(doc)
{
   trace("Received XML: " + doc);
   switch(doc.firstChild.localName)
   {
      case "serverstatus":
         if(STATE == 1)
         {
            updateRooms(doc);
            insertCreateRoomBox();
            STATE = 2;
         }
         else if(STATE == 2)
         {
            updateRooms(doc);
         }
         else
         {
            trace("RECEIVED WRONG MESSAGE: " + STATE);
         }
         _root.LATESTSERVERSTATUS = doc;
         break;
      case "joinroomresult":
         if(STATE == 3)
         {
            if(checkJoinResult(doc.firstChild.attributes.status,doc.firstChild.firstChild.nodeValue))
            {
               STATE = 4;
            }
            else
            {
               STATE = 2;
               _root.ROOMNAME = undefined;
               updateRooms(_root.LATESTSERVERSTATUS);
               insertCreateRoomBox();
            }
         }
         else
         {
            trace("RECEIVED WRONG MESSAGE: " + STATE);
         }
         break;
      case "roomstatus":
         if(STATE == 4)
         {
            updateJoinedRoom(doc);
            insertChatBox();
            insertCreateGameBox();
            insertLeaveRoomButton();
            _root.chatMessagesPanel.insertMessage("Welcome to the TankTrouble chat!","room","System");
            _root.chatMessagesPanel.insertMessage("Activate the chat by pressing Enter. Type /h for further help","room","System");
            STATE = 5;
         }
         else if(STATE == 5)
         {
            updateJoinedRoom(doc);
         }
         else
         {
            trace("RECEIVED WRONG MESSAGE: " + STATE);
         }
         _root.LATESTROOMSTATUS = doc;
         break;
      case "joingameresult":
         if(STATE == 6)
         {
            if(checkJoinResult(doc.firstChild.attributes.result,doc.firstChild.firstChild.nodeValue))
            {
               STATE = 7;
            }
            else
            {
               STATE = 5;
               _root.GAMENAME = undefined;
               _root.GAMEOWNEDBYME = false;
               updateJoinedRoom(_root.LATESTROOMSTATUS);
               insertCreateGameBox();
               insertLeaveRoomButton();
            }
         }
         else
         {
            trace("RECEIVED WRONG MESSAGE: " + STATE);
         }
         break;
      case "gamestatus":
         if(STATE != 7)
         {
            trace("RECEIVED WRONG MESSAGE: " + STATE);
         }
         break;
      case "rounddata":
         setupMaze(doc);
         break;
      case "loginresult":
         if(checkJoinResult(doc.firstChild.attributes.status,doc.firstChild.firstChild.nodeValue))
         {
            _root.AUTHENTICATED = true;
         }
         else
         {
            displayErrorMessage("Server error. Please wait a while and restart the game.");
         }
         break;
      case "player":
         updatePlayerMetaInfo(doc);
         break;
      case "roomchat":
         _root.chatMessagesPanel.insertMessage(doc.firstChild.firstChild.nodeValue,"room",doc.firstChild.attributes.sender);
         break;
      case "gamechat":
         _root.chatMessagesPanel.insertMessage(doc.firstChild.firstChild.nodeValue,"game",doc.firstChild.attributes.sender);
         break;
      case "privatechat":
         _root.chatMessagesPanel.insertMessage(doc.firstChild.firstChild.nodeValue,"private",doc.firstChild.attributes.sender);
         _root.PREVIOUS_PRIVATE_SENDER = doc.firstChild.attributes.sender;
         break;
      default:
         trace("UNKNOWN MESSAGE RECEIVED: " + STATE);
   }
};
if(!connection.connect(HOST,PORT))
{
   displayErrorMessage("Couldn\'t connect to game server. Please wait a while and restart the game.");
}
