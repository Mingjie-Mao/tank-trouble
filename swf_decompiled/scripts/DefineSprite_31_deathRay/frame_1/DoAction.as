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
function targetClosestTank(s, d)
{
   circleCounter = 0;
   straightLine = true;
   targetPosition = undefined;
   var _loc13_ = 1.0000000000000001e+33;
   var _loc9_ = 0;
   while(_loc9_ < _root.TANKS)
   {
      var _loc8_ = _root.game["tank" + _loc9_];
      if(_loc8_ != owner)
      {
         if(_loc8_.alive)
         {
            var _loc4_ = new Object({x:_loc8_.x - this._x - s.x,y:_loc8_.y - this._y - s.y});
            var _loc6_ = new Object({x:d.y,y:- d.x});
            if(d.x * _loc4_.x + d.y * _loc4_.y > 0)
            {
               var _loc12_ = _loc4_.x * _loc4_.x + _loc4_.y * _loc4_.y;
               if(_loc12_ <= _loc13_)
               {
                  _loc13_ = _loc12_;
                  targetPosition = {x:_loc8_.x - this._x,y:_loc8_.y - this._y};
                  straightLine = Math.abs(_loc6_.x * _loc4_.x + _loc6_.y * _loc4_.y) <= 0.00001;
                  if(!straightLine)
                  {
                     var _loc11_ = Math.sqrt(_loc6_.x * _loc6_.x + _loc6_.y * _loc6_.y);
                     _loc6_.x /= _loc11_;
                     _loc6_.y /= _loc11_;
                     var _loc7_ = (_loc4_.x * _loc4_.x + _loc4_.y * _loc4_.y) / (2 * _loc6_.x * _loc4_.x + 2 * _loc6_.y * _loc4_.y);
                     if(_loc7_ > 0)
                     {
                        _loc7_ = Math.max(650 * (_root.SCALE / 50),_loc7_);
                        leftSide = true;
                     }
                     else
                     {
                        _loc7_ = Math.min(-650 * (_root.SCALE / 50),_loc7_);
                        leftSide = false;
                     }
                     var _loc5_ = new Object({x:s.x + _loc6_.x * _loc7_,y:s.y + _loc6_.y * _loc7_});
                     targetCircleRadius = Math.sqrt((_loc5_.x - s.x) * (_loc5_.x - s.x) + (_loc5_.y - s.y) * (_loc5_.y - s.y));
                     targetCircleCenter = _loc5_;
                     if(- (_loc5_.x - s.x) != 0)
                     {
                        if(- (_loc5_.x - s.x) > 0)
                        {
                           targetCircleStartAngle = 1.5707963267948966 + Math.atan((- (_loc5_.y - s.y)) / (- (_loc5_.x - s.x)));
                        }
                        else
                        {
                           targetCircleStartAngle = -1.5707963267948966 + Math.atan((- (_loc5_.y - s.y)) / (- (_loc5_.x - s.x)));
                        }
                     }
                     else if(- (_loc5_.y - s.y) > 0)
                     {
                        targetCircleStartAngle = 3.141592653589793;
                     }
                     else if(- (_loc5_.y - s.y) < 0)
                     {
                        targetCircleStartAngle = 0;
                     }
                     targetCircleStartAngle -= 1.5707963267948966;
                  }
               }
            }
         }
      }
      _loc9_ = _loc9_ + 1;
   }
}
trace("-----");
var collisionPoints = new Array();
var rayThickness = 0;
var ballSize = 0;
var particleCounter = 0;
var ownerColor = 16711680 * owner.turretColor.r / 255 + 65280 * owner.turretColor.g / 255 + 255 * owner.turretColor.b / 255;
var darkerOwnerColor = ownerColor;
var colors = new Array(16777215,ownerColor,16777215,ownerColor,16777215,ownerColor,16777215,ownerColor,16777215,ownerColor);
var alphas = new Array(100,100,100,100,100,100,100,100,100,100);
var fractions = new Array(0,25,50,75,100,125,150,175,200,225);
var nextColor = 16777215;
var lengthCounter = 0;
var curvePoints = new Array();
var circleCounter = 0;
var targetCircleCenter;
var targetCircleRadius;
var targetCircleStartAngle;
var targetPosition;
var straightLine;
var leftSide;
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
   if(warmup == 0)
   {
      _root.soundDeathRayCharge.stop("soundDeathRayCharge");
      if(_root.soundOn)
      {
         _root.soundDeathRayFire.start();
      }
      x = 0;
      y = 0;
      var _loc14_ = new Object({x:x,y:y});
      var _loc13_ = new Object({x:xSpeed,y:ySpeed});
      targetClosestTank(_loc14_,_loc13_);
      var _loc10_ = false;
      while(hitCheck(_root.game.mazebg,{x:x,y:y}))
      {
         var _loc16_ = new Object({x:targetPosition.x - x,y:targetPosition.y - y});
         var _loc4_ = new Object();
         if(straightLine)
         {
            _loc4_.x = xSpeed;
            _loc4_.y = ySpeed;
         }
         else
         {
            var _loc11_ = Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
            _loc4_.x = Math.sin(targetCircleStartAngle + circleCounter * (!leftSide ? 1 : -1) * _loc11_ / targetCircleRadius) * _loc11_;
            _loc4_.y = (- Math.cos(targetCircleStartAngle + circleCounter * (!leftSide ? 1 : -1) * _loc11_ / targetCircleRadius)) * _loc11_;
            if(!leftSide)
            {
               _loc4_.x = - _loc4_.x;
               _loc4_.y = - _loc4_.y;
            }
         }
         xSpeed = _loc4_.x;
         ySpeed = _loc4_.y;
         x += xSpeed;
         y += ySpeed;
         lengthCounter++;
         circleCounter++;
         curvePoints.push({x:x,y:y});
         if(targetPosition != undefined && _loc4_.x * _loc16_.x + _loc4_.y * _loc16_.y <= 0)
         {
            targetClosestTank({x:x,y:y},_loc4_);
         }
         if(hitCheck(_root.game.mazemc,{x:x,y:y}))
         {
            if(!_loc10_)
            {
               var _loc9_ = {x:x,y:y};
               localToGlobal(_loc9_);
               _root.game.mazebg.globalToLocal(_loc9_);
               collisionPoints.push(_loc9_);
               _loc10_ = true;
            }
         }
         else if(_loc10_)
         {
            _loc9_ = {x:x,y:y};
            localToGlobal(_loc9_);
            _root.game.mazebg.globalToLocal(_loc9_);
            collisionPoints.push(_loc9_);
            _loc10_ = false;
         }
      }
      x = curvePoints[curvePoints.length - 1].x;
      y = curvePoints[curvePoints.length - 1].y;
      while(!hitCheck(_root.game.mazebg,{x:x + 3 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x - 3 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x,y:y + 3 * (_root.SCALE / 50)}) || !hitCheck(_root.game.mazebg,{x:x,y:y - 3 * (_root.SCALE / 50)}))
      {
         curvePoints.pop();
         x = curvePoints[curvePoints.length - 1].x;
         y = curvePoints[curvePoints.length - 1].y;
         lengthCounter--;
      }
      active = true;
      warmup--;
   }
   if(active)
   {
      ballSize = Math.max(0,ballSize - 5);
      rayThickness = Math.min(5,rayThickness + 1);
      var _loc8_ = 0;
      while(_loc8_ < collisionPoints.length)
      {
         var _loc5_ = 0;
         while(_loc5_ < 1)
         {
            if(Math.random() <= 0.5)
            {
               p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
               this.swapDepths(p);
               particleCounter++;
               var _loc12_ = Math.random() * 360;
               _loc11_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
               p.x = collisionPoints[_loc8_].x;
               p.y = collisionPoints[_loc8_].y;
               p._x = p.x;
               p._y = p.y;
               p.lineStyle((Math.random() + 1) * (_root.SCALE / 50),Math.random() <= 0.7000000000000001 ? ownerColor : darkerOwnerColor);
               p.moveTo(0,0);
               var _loc7_ = 0;
               var _loc6_ = 0;
               _loc5_ = 0;
               while(_loc5_ < 8)
               {
                  _loc7_ += Math.random() * 10 - 5;
                  _loc6_ += Math.random() * 10 - 5;
                  p.lineTo(_loc7_ * (_root.SCALE / 50),_loc6_ * (_root.SCALE / 50));
                  _loc5_ = _loc5_ + 1;
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
            _loc5_ = _loc5_ + 1;
         }
         _loc8_ = _loc8_ + 1;
      }
      _loc8_ = 0;
      while(_loc8_ < collisionPoints.length)
      {
         if(Math.random() <= 0.5)
         {
            p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
            this.swapDepths(p);
            particleCounter++;
            _loc12_ = Math.random() * 360;
            _loc11_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
            p.x = collisionPoints[_loc8_].x;
            p.y = collisionPoints[_loc8_].y;
            p._x = p.x;
            p._y = p.y;
            p.lineStyle((Math.random() * 2 + 1) * (_root.SCALE / 50),5066061);
            p.moveTo(0,0);
            p.lineTo(1,0);
            p.xspeed = Math.cos(_loc12_) * _loc11_;
            p.yspeed = Math.sin(_loc12_) * _loc11_;
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
         _loc8_ = _loc8_ + 1;
      }
      _loc8_ = 0;
      while(_loc8_ < _root.TANKS)
      {
         var _loc3_ = _root.game["tank" + _loc8_];
         if(_loc3_ != owner)
         {
            if(_loc3_.alive)
            {
               _loc5_ = 0;
               while(_loc5_ < _loc3_.hitPointsFront.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsFront[_loc5_].x,y:_loc3_.hitPointsFront[_loc5_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc8_);
                     break;
                  }
                  _loc5_ = _loc5_ + 1;
               }
               _loc5_ = 0;
               while(_loc5_ < _loc3_.hitPointsLeft.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsLeft[_loc5_].x,y:_loc3_.hitPointsLeft[_loc5_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc8_);
                     break;
                  }
                  _loc5_ = _loc5_ + 1;
               }
               _loc5_ = 0;
               while(_loc5_ < _loc3_.hitPointsRear.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsRear[_loc5_].x,y:_loc3_.hitPointsRear[_loc5_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc8_);
                     break;
                  }
                  _loc5_ = _loc5_ + 1;
               }
            }
         }
         _loc8_ = _loc8_ + 1;
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
      if(nextColor == ownerColor)
      {
         nextColor = 16777215;
      }
      else
      {
         nextColor = ownerColor;
      }
   }
   _loc8_ = 0;
   while(_loc8_ < fractions.length)
   {
      fractions[_loc8_] -= 5;
      _loc8_ = _loc8_ + 1;
   }
   clear();
   lineStyle(rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation + 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   _loc8_ = 0;
   while(_loc8_ < curvePoints.length)
   {
      lineTo(curvePoints[_loc8_].x,curvePoints[_loc8_].y);
      _loc8_ = _loc8_ + 1;
   }
   lineStyle(0.5 * rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation - 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   _loc8_ = 0;
   while(_loc8_ < curvePoints.length)
   {
      lineTo(curvePoints[_loc8_].x,curvePoints[_loc8_].y);
      _loc8_ = _loc8_ + 1;
   }
   if(active)
   {
      lifetime--;
   }
   if(lifetime == 0)
   {
      active = false;
   }
};
