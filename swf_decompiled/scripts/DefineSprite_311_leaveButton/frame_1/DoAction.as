glowAlpha = 0;
targetScale = 100;
scaleSpeed = 0;
removeCount = 5;
useHandCursor = true;
onRelease = function()
{
};
tabEnabled = false;
onEnterFrame = function()
{
   if(this.hitTest(_root._xmouse,_root._ymouse,true))
   {
      glowAlpha = Math.min(1,glowAlpha + 0.1);
   }
   else
   {
      glowAlpha = Math.max(0,glowAlpha - 0.1);
   }
   var _loc3_ = 8947967;
   var _loc4_ = new flash.filters.GlowFilter(_loc3_,glowAlpha,5,5,3,3,false,false);
   this.filters = new Array(_loc4_);
   scaleSpeed += (targetScale - _xscale) * 0.2;
   scaleSpeed *= 0.7000000000000001;
   _xscale = _xscale + scaleSpeed;
   _yscale = _xscale;
   if(targetScale == 0 && _xscale < 3)
   {
      removeCount--;
      this._visible = false;
   }
   if(removeCount <= 0)
   {
      this.removeMovieClip();
   }
};
