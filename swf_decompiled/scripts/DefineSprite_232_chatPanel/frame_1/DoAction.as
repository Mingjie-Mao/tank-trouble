function activatePrivateChat(receiver)
{
   if(!active)
   {
      active = true;
   }
   this.message.text = "";
   Selection.setFocus(message);
   messageType = "private";
   messageReceiver = receiver;
}
targetScale = 0;
scaleSpeed = 0;
removeCount = 5;
active = false;
pressedEarlier = false;
messageType = "room";
messageReceiver = "";
remove = false;
var curseWords = new Array("fuck","shit","bitch","cunt");
message.onSetFocus = function(oldFocus)
{
   Key.removeListener(_root.settings);
   Key.removeListener(_root.sound);
};
message.onKillFocus = function(newFocus)
{
   Key.addListener(_root.settings);
   Key.addListener(_root.sound);
};
onEnterFrame = function()
{
   if(active && !remove)
   {
      targetScale = 100;
   }
   else
   {
      targetScale = 0;
   }
   scaleSpeed += (targetScale - _xscale) * 0.2;
   scaleSpeed *= 0.7000000000000001;
   _xscale = _xscale + scaleSpeed;
   _yscale = _xscale;
   if(_xscale < 0)
   {
      _xscale = 0;
      _yscale = _xscale;
      scaleSpeed = 0;
   }
   if(targetScale == 0 && remove && _xscale < 3)
   {
      removeCount--;
      this._visible = false;
   }
   if(removeCount <= 0)
   {
      this.removeMovieClip();
   }
};
message.onChanged = function(changedField)
{
   if(active)
   {
      if(changedField.text.substr(0,3) == "/r ")
      {
         if(_root.PREVIOUS_PRIVATE_SENDER != undefined)
         {
            messageType = "private";
            messageReceiver = _root.PREVIOUS_PRIVATE_SENDER;
            changedField.text = changedField.text.slice(3);
         }
      }
      if(changedField.text.substr(0,3) == "/g ")
      {
         if(_root.GAMENAME != undefined)
         {
            messageType = "game";
            changedField.text = changedField.text.slice(3);
         }
      }
      if(changedField.text.substr(0,3) == "/w ")
      {
         if(changedField.text.indexOf(" ",3) > 0)
         {
            var _loc3_ = changedField.text.substring(3,changedField.text.indexOf(" ",3));
            if(_root.getMemberWithName(_loc3_) != null && _root.loginInfo.p1n != _loc3_)
            {
               messageType = "private";
               messageReceiver = _loc3_;
               changedField.text = changedField.text.slice(3 + _loc3_.length + 1);
            }
         }
      }
   }
};
onKeyDown = function()
{
   if(Key.isDown(13) && !pressedEarlier)
   {
      pressedEarlier = true;
      active = !active;
      if(active)
      {
         this.message.text = "";
         Selection.setFocus(message);
         messageType = "room";
         messageReceiver = "";
      }
      else
      {
         var _loc3_ = this.message.text.trim();
         if(length(_loc3_) > 0)
         {
            var _loc6_ = _loc3_.toLowerCase();
            var _loc5_ = -1;
            var _loc4_ = 0;
            while(_loc4_ < curseWords.length)
            {
               while((_loc5_ = _loc6_.indexOf(curseWords[_loc4_])) >= 0)
               {
                  _loc3_ = _loc3_.slice(0,_loc5_) + "%@#$" + _loc3_.slice(_loc5_ + length(curseWords[_loc4_]));
                  _loc6_ = _loc3_.toLowerCase();
               }
               _loc4_ = _loc4_ + 1;
            }
            if(_loc3_ == "/h")
            {
               _root.chatMessagesPanel.insertMessage("Type \'/g \' to send a game message","room","System");
               _root.chatMessagesPanel.insertMessage("Type \'/w <username> \' to send a private message","room","System");
               _root.chatMessagesPanel.insertMessage("Type \'/r \' to reply to a private message","room","System");
               return undefined;
            }
            switch(messageType)
            {
               case "room":
                  trace("ROOM CHAT: " + _loc3_);
                  var message = new XML();
                  var _loc7_ = new XMLNode(1,"roomchat");
                  _loc7_.appendChild(new XMLNode(2,_loc3_));
                  message.appendChild(_loc7_);
                  _root.connection.send(message);
                  break;
               case "private":
                  trace("PRIVATE CHAT TO " + messageReceiver + ": " + _loc3_);
                  var message = new XML();
                  _loc7_ = new XMLNode(1,"privatechat");
                  _loc7_.appendChild(new XMLNode(2,_loc3_));
                  _loc7_.attributes.recipient = messageReceiver;
                  message.appendChild(_loc7_);
                  _root.connection.send(message);
                  break;
               case "game":
                  trace("GAME CHAT: " + _loc3_);
                  var message = new XML();
                  _loc7_ = new XMLNode(1,"gamechat");
                  _loc7_.appendChild(new XMLNode(2,_loc3_));
                  message.appendChild(_loc7_);
                  _root.connection.send(message);
            }
         }
         Selection.setFocus(null);
      }
   }
};
onKeyUp = function()
{
   if(!Key.isDown(13))
   {
      pressedEarlier = false;
   }
};
Key.addListener(this);
