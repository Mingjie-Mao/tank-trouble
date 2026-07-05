onClipEvent(load){
   function toggleSettingsPanel()
   {
      _root.settingsPanelRequested = !_root.settingsPanelRequested;
      this.gotoAndStop(1 + _root.settingsPanelRequested);
   }
   this.settingsButton.tabEnabled = false;
   _root.settingsPanelRequested = true;
   toggleSettingsPanel();
   onMouseDown = function()
   {
      if(this.hitTest(_root._xmouse,_root._ymouse,true))
      {
         this.toggleSettingsPanel();
      }
   };
   onKeyDown = function()
   {
      if(Key.isDown(32) && !pressedEarlier)
      {
         this.toggleSettingsPanel();
         pressedEarlier = true;
      }
   };
   onKeyUp = function()
   {
      if(!Key.isDown(32))
      {
         pressedEarlier = false;
      }
   };
   Key.addListener(this);
}
