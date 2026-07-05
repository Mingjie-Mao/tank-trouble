onClipEvent(load){
   function toggleSound()
   {
      _root.soundOn = !_root.soundOn;
      this.gotoAndStop(1 + _root.soundOn);
   }
   this.soundButton.tabEnabled = false;
   _root.soundOn = false;
   toggleSound();
   onMouseDown = function()
   {
      if(this.hitTest(_root._xmouse,_root._ymouse,true))
      {
         this.toggleSound();
      }
   };
   onKeyDown = function()
   {
      if(Key.isDown(65) && !pressedEarlier)
      {
         this.toggleSound();
         pressedEarlier = true;
      }
   };
   onKeyUp = function()
   {
      if(!Key.isDown(65))
      {
         pressedEarlier = false;
      }
   };
   Key.addListener(this);
}
