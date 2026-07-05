function repulseBullet(bullet)
{
   var _loc5_ = {x:bullet.x,y:bullet.y};
   _root.game.localToGlobal(_loc5_);
   if(hitTest(_loc5_.x,_loc5_.y,true) && targetSize > 0)
   {
      var _loc4_ = {x:_X - bullet.x,y:_Y - bullet.y};
      var _loc8_ = _loc4_.x * bullet.xSpeed + _loc4_.y * bullet.ySpeed;
      if(_loc8_ > 0)
      {
         bullet.xSpeed = - bullet.xSpeed;
         bullet.ySpeed = - bullet.ySpeed;
         var _loc3_ = {x:_loc4_.y,y:- _loc4_.x};
         var _loc7_ = _loc3_.x * _loc3_.x + _loc3_.y * _loc3_.y;
         var _loc6_ = (bullet.xSpeed * _loc3_.x + bullet.ySpeed * _loc3_.y) / _loc7_;
         bullet.xSpeed -= 2 * _loc3_.x * _loc6_;
         bullet.ySpeed -= 2 * _loc3_.y * _loc6_;
         targetSize -= 10;
         layers[hitCount].x = - _loc4_.x;
         layers[hitCount].y = - _loc4_.y;
         hitCount++;
         shieldGraphic.impact(_X - _loc4_.x,_Y - _loc4_.y,hitCount);
      }
   }
}
var shieldGraphic = _root.game.mazebg.attachMovie("shieldGraphic","shieldGraphic-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
shieldGraphic.owner = owner;
shieldGraphic.shield = this;
var size = 0;
var waveColors = new Array(16777215,16777215,16777215,16777215,16777215);
var waveAlphas = new Array(0,0,80,0,0);
var waveFractions = new Array(0,1,2,3,250);
var layerColors = new Array(shieldColor,shieldColor,shieldColor,shieldColor,shieldColor);
var layerAlphas = new Array(0,0,70,30,30);
var layers = new Array();
var hitCount = 0;
var outside = 2 * _root.SHIELDSIZE * (_root.SCALE / 50);
layers.push({x:outside,y:outside,fractions:new Array(0,1,2,3,250)});
layers.push({x:outside,y:outside,fractions:new Array(0,1,2,3,250)});
onEnterFrame = function()
{
   if(_root.frozen)
   {
      return undefined;
   }
   if(!owner.alive)
   {
      owner.equipment = undefined;
      owner.currentEquipment = "";
      this.removeMovieClip();
   }
   clear();
   lineStyle(size * (_root.SCALE / 50));
   var _loc3_ = 0;
   while(_loc3_ < layers.length)
   {
      if(_loc3_ < hitCount)
      {
         if(layers[_loc3_].fractions[1] <= 140)
         {
            layers[_loc3_].fractions[3] += 15;
            if(layers[_loc3_].fractions[3] > 18)
            {
               layers[_loc3_].fractions[2] += 15;
            }
            if(layers[_loc3_].fractions[2] > 17)
            {
               layers[_loc3_].fractions[1] += 15;
            }
         }
      }
      lineGradientStyle("radial",layerColors,layerAlphas,layers[_loc3_].fractions,{matrixType:"box",x:-2 * _root.SHIELDSIZE * (_root.SCALE / 50) + layers[_loc3_].x,y:-2 * _root.SHIELDSIZE * (_root.SCALE / 50) + layers[_loc3_].y,w:4 * _root.SHIELDSIZE * (_root.SCALE / 50),h:4 * _root.SHIELDSIZE * (_root.SCALE / 50),r:0});
      moveTo(0,0);
      lineTo(1,0);
      _loc3_ = _loc3_ + 1;
   }
   waveFractions[3] += 2;
   if(waveFractions[3] > 15)
   {
      waveFractions[2] += 2;
   }
   if(waveFractions[2] > 15)
   {
      waveFractions[1] += 2;
   }
   if(waveFractions[1] > 60)
   {
      waveFractions[3] = 3;
      waveFractions[2] = 2;
      waveFractions[1] = 1;
   }
   lineGradientStyle("radial",waveColors,waveAlphas,waveFractions,{matrixType:"box",x:-2 * _root.SHIELDSIZE * (_root.SCALE / 50),y:-2 * _root.SHIELDSIZE * (_root.SCALE / 50),w:4 * _root.SHIELDSIZE * (_root.SCALE / 50),h:4 * _root.SHIELDSIZE * (_root.SCALE / 50),r:0});
   moveTo(0,0);
   lineTo(1,0);
   if(hitCount >= 2)
   {
      targetSize = 0;
      waveAlphas[2] = Math.max(waveAlphas[2] - 10,0);
   }
   if(targetSize > size)
   {
      size += 2;
   }
   if(targetSize < size)
   {
      size -= 2;
   }
   if(targetSize == 0)
   {
      size -= 2;
   }
   if(size <= 0)
   {
      owner.equipment = undefined;
      owner.currentEquipment = "";
      this.removeMovieClip();
   }
};
