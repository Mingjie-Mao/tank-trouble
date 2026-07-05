function insertMessage(message, type, sender)
{
   var _loc2_ = this.attachMovie("chatMessagePanel","chatMessagePanel" + this.getNextHighestDepth(),this.getNextHighestDepth());
   _loc2_.x = 0;
   _loc2_.y = - messageHeight + (messages[0] == undefined ? 0 : messages[0].y);
   _loc2_._x = _loc2_.x;
   _loc2_._y = _loc2_.y;
   _loc2_._alpha = 0;
   _loc2_.message.text = sender + " says \'" + message + "\'";
   _loc2_.type.gotoAndStop(type);
   if(type == "private")
   {
      _loc2_.message.textColor = 16711935;
   }
   _loc2_.lifeTime = 300;
   messages.unshift(_loc2_);
}
var messages = new Array();
var messageHeight = 20;
var remove = false;
var removeCount = 100;
onEnterFrame = function()
{
   if(remove)
   {
      removeCount -= 10;
      var _loc2_ = 0;
      while(_loc2_ < messages.length)
      {
         messages[_loc2_]._alpha = Math.min(removeCount,messages[_loc2_]._alpha);
         _loc2_ = _loc2_ + 1;
      }
      if(removeCount <= 0)
      {
         this.removeMovieClip();
      }
   }
   else
   {
      _loc2_ = 0;
      while(_loc2_ < messages.length)
      {
         if(messages[_loc2_].y > -15)
         {
            messages[_loc2_].lifeTime--;
            if(messages[_loc2_].lifeTime > 0)
            {
               messages[_loc2_]._alpha = Math.min(100,messages[_loc2_]._alpha + 10);
            }
         }
         if(_loc2_ * messageHeight > messages[_loc2_].y)
         {
            messages[_loc2_].y += 2;
            messages[_loc2_]._y = messages[_loc2_].y;
         }
         _loc2_ = _loc2_ + 1;
      }
      if(messages.length > 0)
      {
         if(messages[messages.length - 1].y > 200)
         {
            messages[messages.length - 1].lifeTime = 0;
         }
         if(messages[messages.length - 1].lifeTime <= 0)
         {
            messages[messages.length - 1]._alpha -= 10;
            if(messages[messages.length - 1]._alpha <= 0)
            {
               messages[messages.length - 1].removeMovieClip();
               messages.pop();
            }
         }
      }
   }
};
