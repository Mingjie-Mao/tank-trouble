function hitCheck(mc, point)
{
   localToGlobal(point);
   if(mc.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
function hitCheck2(mc, point)
{
   mc.localToGlobal(point);
   if(this.hitTest(point.x,point.y,true))
   {
      return true;
   }
   return false;
}
var collisionPoints = new Array();
var rayThickness = 0;
var ballSize = 0;
var particleCounter = 0;
var ownerColor = 16711680 * owner.turretColor.r / 255 + 65280 * owner.turretColor.g / 255 + 255 * owner.turretColor.b / 255;
var darkerOwnerColor = 10027008 * owner.turretColor.r / 255 + 39168 * owner.turretColor.g / 255 + 153 * owner.turretColor.b / 255;
var colors = new Array(0,ownerColor,0,ownerColor,0,ownerColor,0,ownerColor,0,ownerColor);
var alphas = new Array(100,100,100,100,100,100,100,100,100,100);
var fractions = new Array(0,25,50,75,100,125,150,175,200,225);
var nextColor = 0;
var lengthCounter = 0;
onEnterFrame = function()
{
   if(_root.frozen)
   {
      _root.soundDeathRayCharge.stop("soundDeathRayCharge");
      _root.soundDeathRayFire.stop("soundDeathRayFire");
      return undefined;
   }
   if(!owner.alive)
   {
      _root.soundDeathRayCharge.stop("soundDeathRayCharge");
      _root.soundDeathRayFire.stop("soundDeathRayFire");
      this.removeMovieClip();
   }
   if(warmup > 0)
   {
      ballSize += 0.75;
      warmup--;
   }
   if(warmup > 4)
   {
   }
   if(warmup == 0)
   {
      _root.soundDeathRayCharge.stop("soundDeathRayCharge");
      if(_root.soundOn)
      {
         _root.soundDeathRayFire.start();
      }
      x = 0;
      y = 0;
      var _loc9_ = false;
      while(hitCheck(_root.game.mazebg,{x:x,y:y}))
      {
         x += xSpeed;
         y += ySpeed;
         lengthCounter++;
         if(hitCheck(_root.game.mazemc,{x:x,y:y}))
         {
            if(!_loc9_)
            {
               var _loc8_ = {x:x,y:y};
               localToGlobal(_loc8_);
               _root.game.mazebg.globalToLocal(_loc8_);
               collisionPoints.push(_loc8_);
               _loc9_ = true;
            }
         }
         else if(_loc9_)
         {
            _loc8_ = {x:x,y:y};
            localToGlobal(_loc8_);
            _root.game.mazebg.globalToLocal(_loc8_);
            collisionPoints.push(_loc8_);
            _loc9_ = false;
         }
      }
      while(!hitCheck(_root.game.mazebg,{x:x + 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x - 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x,y:y + 2.5 * (_root.SCALE / 50)}) || !hitCheck(_root.game.mazebg,{x:x,y:y - 2.5 * (_root.SCALE / 50)}))
      {
         x -= xSpeed;
         y -= ySpeed;
         lengthCounter--;
      }
      active = true;
      warmup--;
   }
   if(active)
   {
      ballSize = Math.max(0,ballSize - 5);
      rayThickness = Math.min(5,rayThickness + 1);
      var _loc7_ = 0;
      while(_loc7_ < collisionPoints.length)
      {
         var _loc4_ = 0;
         while(_loc4_ < 1)
         {
            if(Math.random() <= 0.5)
            {
               p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
               this.swapDepths(p);
               particleCounter++;
               var _loc11_ = Math.random() * 360;
               var _loc10_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
               p.x = collisionPoints[_loc7_].x;
               p.y = collisionPoints[_loc7_].y;
               p._x = p.x;
               p._y = p.y;
               p.lineStyle((Math.random() + 1) * (_root.SCALE / 50),Math.random() <= 0.7000000000000001 ? ownerColor : darkerOwnerColor);
               p.moveTo(0,0);
               var _loc6_ = 0;
               var _loc5_ = 0;
               _loc4_ = 0;
               while(_loc4_ < 8)
               {
                  _loc6_ += Math.random() * 10 - 5;
                  _loc5_ += Math.random() * 10 - 5;
                  p.lineTo(_loc6_ * (_root.SCALE / 50),_loc5_ * (_root.SCALE / 50));
                  _loc4_ = _loc4_ + 1;
               }
               p.onEnterFrame = function()
               {
                  if(_root.frozen)
                  {
                     return undefined;
                  }
                  this.removeMovieClip();
               };
            }
            _loc4_ = _loc4_ + 1;
         }
         _loc7_ = _loc7_ + 1;
      }
      _loc7_ = 0;
      while(_loc7_ < collisionPoints.length)
      {
         if(Math.random() <= 0.5)
         {
            p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
            this.swapDepths(p);
            particleCounter++;
            _loc11_ = Math.random() * 360;
            _loc10_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
            p.x = collisionPoints[_loc7_].x;
            p.y = collisionPoints[_loc7_].y;
            p._x = p.x;
            p._y = p.y;
            p.lineStyle((Math.random() * 2 + 1) * (_root.SCALE / 50),5066061);
            p.moveTo(0,0);
            p.lineTo(1,0);
            p.xspeed = Math.cos(_loc11_) * _loc10_;
            p.yspeed = Math.sin(_loc11_) * _loc10_;
            p.lifetime = 12;
            p.alpha = 100;
            p.onEnterFrame = function()
            {
               if(_root.frozen)
               {
                  return undefined;
               }
               this.x += this.xspeed;
               this.y += this.yspeed;
               this._x = this.x;
               this._y = this.y;
               this._alpha = this.alpha;
               this.xspeed *= 0.9500000000000001;
               this.yspeed *= 0.9500000000000001;
               this.lifetime = this.lifetime - 1;
               if(this.lifetime <= 0)
               {
                  this.alpha -= 25;
               }
               if(this.alpha <= 0)
               {
                  this.removeMovieClip();
               }
            };
         }
         _loc7_ = _loc7_ + 1;
      }
      _loc7_ = 0;
      while(_loc7_ < _root.TANKS)
      {
         var _loc3_ = _root.game["tank" + _loc7_];
         if(_loc3_ != owner)
         {
            if(_loc3_.alive)
            {
               _loc4_ = 0;
               while(_loc4_ < _loc3_.hitPointsFront.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsFront[_loc4_].x,y:_loc3_.hitPointsFront[_loc4_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc7_);
                     break;
                  }
                  _loc4_ = _loc4_ + 1;
               }
               _loc4_ = 0;
               while(_loc4_ < _loc3_.hitPointsLeft.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsLeft[_loc4_].x,y:_loc3_.hitPointsLeft[_loc4_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc7_);
                     break;
                  }
                  _loc4_ = _loc4_ + 1;
               }
               _loc4_ = 0;
               while(_loc4_ < _loc3_.hitPointsRear.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsRear[_loc4_].x,y:_loc3_.hitPointsRear[_loc4_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc7_);
                     break;
                  }
                  _loc4_ = _loc4_ + 1;
               }
            }
         }
         _loc7_ = _loc7_ + 1;
      }
   }
   if(!active && warmup < 0)
   {
      rayThickness -= 3;
      if(rayThickness <= 0)
      {
         owner.deathRayReady = true;
         _root.setWeapon(owner,"bullet");
         this.removeMovieClip();
      }
   }
   if(fractions[0] <= 0)
   {
      fractions.shift();
      fractions.push(250);
      colors.shift();
      colors.push(nextColor);
      nextColor = ownerColor - nextColor;
   }
   _loc7_ = 0;
   while(_loc7_ < fractions.length)
   {
      fractions[_loc7_] -= 5;
      _loc7_ = _loc7_ + 1;
   }
   clear();
   lineStyle(rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation + 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   lineTo(x,y);
   lineStyle(0.5 * rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation - 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   lineTo(x,y);
   if(active)
   {
      lifetime--;
   }
   if(lifetime == 0)
   {
      active = false;
   }
};
