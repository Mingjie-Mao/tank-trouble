onClipEvent(mouseUp){
   if(this.hitTest(_root._xmouse,_root._ymouse,false))
   {
      _parent.laserSlider.activate = true;
      _parent.fragSlider.activate = true;
      _parent.gatlingSlider.activate = true;
      _parent.homingSlider.activate = true;
      _parent.deathRaySlider.activate = true;
      _parent.randomMazeSlider.activate = true;
      _parent.myOwnMazeSlider.activate = true;
      _parent.othersMazeSlider.activate = false;
      _parent.newMouseControlSlider.activate = false;
   }
}
