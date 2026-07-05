function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(active)
   {
      i = 0;
      while(i < _root.FRAGHITCHECKINTERVALS)
      {
         x += xSpeed;
         y += ySpeed;
         _X = x;
         _Y = y;
         if(hitCheck(_root.game.mazemc,{x:0,y:0}))
         {
            if(_root.soundOn)
            {
               if(Math.random() > 0.5)
               {
                  _root.soundFragmentHit.start();
               }
               else
               {
                  _root.soundFragmentHit2.start();
               }
            }
            active = false;
            i = _root.FRAGHITCHECKINTERVALS;
         }
         i++;
      }
      _X = x;
      _Y = y;
      _rotation = _rotation + rotSpeed;
      var i = 0;
      while(i < _root.TANKS)
      {
         if(_root.game["tank" + i].alive && hitCheck(_root.game["tank" + i],{x:0,y:0}))
         {
            _root.registerHit(owner,_root.game["tank" + i]);
            _root.destroyTank(i);
            this.removeMovieClip();
         }
         i++;
      }
   }
   else
   {
      _alpha = _alpha - 5;
      if(_alpha <= 0)
      {
         this.removeMovieClip();
      }
   }
};
