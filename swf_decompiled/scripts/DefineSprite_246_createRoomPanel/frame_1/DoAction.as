targetScale = 100;
scaleSpeed = 0;
removeCount = 5;
onEnterFrame = function()
{
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
