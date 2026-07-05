targetScale = 100;
scaleSpeed = 0;
removeCount = _root.NUMBEROFFRAMESBEFORERESET;
onEnterFrame = function()
{
   scaleSpeed += (targetScale - _xscale) * 2;
   scaleSpeed *= 0.2;
   _xscale = _xscale + scaleSpeed;
   _yscale = _xscale;
   if(!randomMazeSlider.activate && !myOwnMazeSlider.activate && !othersMazeSlider.activate)
   {
      randomMazeSlider.activate = true;
   }
   if(_root.AIEnabled)
   {
      if(laserSlider.activate)
      {
         laserSlider.activate = false;
      }
      if(fragSlider.activate)
      {
         fragSlider.activate = false;
      }
      if(gatlingSlider.activate)
      {
         gatlingSlider.activate = false;
      }
      if(homingSlider.activate)
      {
         homingSlider.activate = false;
      }
      if(deathRaySlider.activate)
      {
         deathRaySlider.activate = false;
      }
   }
   if(targetScale == 0 && _xscale < 3)
   {
      removeCount--;
      this._visible = false;
   }
   if(removeCount <= 0)
   {
      _root.setupBattle();
      this.removeMovieClip();
   }
};
