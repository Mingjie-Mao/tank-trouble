onClipEvent(mouseUp){
   if(this.hitTest(_root._xmouse,_root._ymouse,false))
   {
      if(_root.AIEnabled)
      {
         if(_parent.laserSlider.activate)
         {
            _parent.laserSlider.activate = false;
         }
         if(_parent.fragSlider.activate)
         {
            _parent.fragSlider.activate = false;
         }
         if(_parent.gatlingSlider.activate)
         {
            _parent.gatlingSlider.activate = false;
         }
         if(_parent.homingSlider.activate)
         {
            _parent.homingSlider.activate = false;
         }
         if(_parent.deathRaySlider.activate)
         {
            _parent.deathRaySlider.activate = false;
         }
      }
      var newActiveWeapons = new Array();
      if(_parent.laserSlider.activate)
      {
         newActiveWeapons.push("laser");
      }
      if(_parent.fragSlider.activate)
      {
         newActiveWeapons.push("frag");
      }
      if(_parent.gatlingSlider.activate)
      {
         newActiveWeapons.push("gatling");
      }
      if(_parent.homingSlider.activate)
      {
         newActiveWeapons.push("homing");
      }
      if(_parent.deathRaySlider.activate)
      {
         newActiveWeapons.push("deathRay");
      }
      _root.settingsActiveWeapons = newActiveWeapons;
      _root.settingsPlayRandomMazes = _parent.randomMazeSlider.activate;
      _root.settingsPlayMyCustomMazes = _parent.myOwnMazeSlider.activate;
      _root.settingsPlayOtherCustomMazes = _parent.othersMazeSlider.activate;
      _root.settingsUseNewMouseControl = _parent.newMouseControlSlider.activate;
      if(!_parent.wasMouseShowing)
      {
         Mouse.hide();
         _root.scopeCross._alpha = 100;
         _root.scopeCircle._alpha = 100;
      }
      _root.frozen = false;
      _parent.targetScale = 0;
   }
}
