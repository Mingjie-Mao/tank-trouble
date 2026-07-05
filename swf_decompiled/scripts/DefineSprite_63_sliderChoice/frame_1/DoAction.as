ON = -21;
OFF = 0;
clicked = false;
grab = false;
slide = false;
sound = new Sound(this);
sound.attachSound("click");
if(activate)
{
   slider._x = ON;
}
else
{
   slider._x = OFF;
}
onMouseDown = function()
{
   if(slider.hitTest(_root._xmouse,_root._ymouse,true))
   {
      clicked = true;
      grab = true;
      startX = _root._xmouse;
      xOffset = slider._x - _root._xmouse;
   }
   else if(hitzone.hitTest(_root._xmouse,_root._ymouse,true))
   {
      clicked = true;
   }
};
onMouseUp = function()
{
   if(clicked && !slide)
   {
      activate = !activate;
   }
   else if(slide)
   {
      if(slider._x < ON + (OFF - ON) / 2)
      {
         activate = true;
      }
      else
      {
         activate = false;
      }
   }
   clicked = false;
   grab = false;
   slide = false;
};
onMouseMove = function()
{
   if(grab && Math.abs(startX - _root._xmouse) > 2)
   {
      slide = true;
   }
};
onEnterFrame = function()
{
   oldX = slider._x;
   if(!grab)
   {
      if(activate)
      {
         slider._x += (ON - slider._x) * 0.8;
      }
      else
      {
         slider._x += (OFF - slider._x) * 0.8;
      }
   }
   if(slide)
   {
      slider._x = Math.min(OFF,Math.max(ON,_root._xmouse + xOffset));
   }
   if(slider._x < ON + 1)
   {
      slider._x = ON;
   }
   if(slider._x > OFF - 1)
   {
      slider._x = OFF;
   }
   color._x = slider._x + slider._width / 2;
   if(oldX != ON && slider._x == ON || oldX != OFF && slider._x == OFF)
   {
      if(_root.soundOn)
      {
         sound.start();
      }
      callback(slider._x == ON);
   }
};
