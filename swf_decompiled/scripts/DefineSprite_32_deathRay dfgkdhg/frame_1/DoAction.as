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
var darkerOwnerColor = ownerColor;
var colors = new Array(16777215,ownerColor,16777215,ownerColor,16777215,ownerColor,16777215,ownerColor,16777215,ownerColor);
var alphas = new Array(100,100,100,100,100,100,100,100,100,100);
var fractions = new Array(0,25,50,75,100,125,150,175,200,225);
var nextColor = 16777215;
var lengthCounter = 0;
var biggestCircleCenter;
var biggestCircleRadius;
var biggestCircleStartAngle;
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
      var _loc19_ = new Object({x:0,y:0});
      var _loc17_ = new Object({x:xSpeed * 400,y:ySpeed * 400});
      var _loc5_ = new Object({x:_loc17_.y,y:- _loc17_.x});
      biggestCircleRadius = 0;
      straightLine = true;
      var _loc10_ = 0;
      while(_loc10_ < _root.TANKS)
      {
         var _loc3_ = _root.game["tank" + _loc10_];
         if(_loc3_ != owner)
         {
            if(_loc3_.alive)
            {
               var _loc20_ = new Object({x:_loc3_.x - this._x,y:_loc3_.y - this._y});
               if(_loc17_.x * _loc20_.x + _loc17_.y * _loc20_.y > 0)
               {
                  straightLine = Math.abs(_loc5_.x * _loc20_.x + _loc5_.y * _loc20_.y) <= 0.00001;
                  if(straightLine)
                  {
                     break;
                  }
                  var _loc16_ = Math.sqrt(_loc5_.x * _loc5_.x + _loc5_.y * _loc5_.y);
                  _loc5_.x /= _loc16_;
                  _loc5_.y /= _loc16_;
                  var _loc9_ = (_loc20_.x * _loc20_.x + _loc20_.y * _loc20_.y) / (2 * _loc5_.x * _loc20_.x + 2 * _loc5_.y * _loc20_.y);
                  if(_loc9_ > 0)
                  {
                     _loc9_ = Math.max(650 * (_root.SCALE / 50),_loc9_);
                  }
                  else
                  {
                     _loc9_ = Math.min(-650 * (_root.SCALE / 50),_loc9_);
                  }
                  var _loc4_ = new Object({x:_loc5_.x * _loc9_,y:_loc5_.y * _loc9_});
                  var _loc11_ = Math.sqrt(_loc4_.x * _loc4_.x + _loc4_.y * _loc4_.y);
                  if(_loc11_ > biggestCircleRadius || _loc11_ == biggestCircleRadius && Math.random() >= 0.5)
                  {
                     biggestCircleRadius = _loc11_;
                     biggestCircleCenter = _loc4_;
                     if(- _loc4_.x != 0)
                     {
                        if(- _loc4_.x > 0)
                        {
                           biggestCircleStartAngle = 1.5707963267948966 + Math.atan((- _loc4_.y) / (- _loc4_.x));
                        }
                        else
                        {
                           biggestCircleStartAngle = -1.5707963267948966 + Math.atan((- _loc4_.y) / (- _loc4_.x));
                        }
                     }
                     else if(- _loc4_.y > 0)
                     {
                        biggestCircleStartAngle = 3.141592653589793;
                     }
                     else if(- _loc4_.y < 0)
                     {
                        biggestCircleStartAngle = 0;
                     }
                     biggestCircleStartAngle -= 1.5707963267948966;
                     leftSide = _loc5_.x * _loc20_.x + _loc5_.y * _loc20_.y > 0;
                  }
               }
            }
         }
         _loc10_ = _loc10_ + 1;
      }
      var _loc13_ = false;
      while(hitCheck(_root.game.mazebg,{x:x,y:y}))
      {
         if(straightLine)
         {
            x += xSpeed;
            y += ySpeed;
         }
         else
         {
            var _loc14_ = (!leftSide ? 1 : -1) * Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
            x = biggestCircleCenter.x + Math.cos(biggestCircleStartAngle + lengthCounter * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
            y = biggestCircleCenter.y + Math.sin(biggestCircleStartAngle + lengthCounter * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         }
         lengthCounter++;
         if(hitCheck(_root.game.mazemc,{x:x,y:y}))
         {
            if(!_loc13_)
            {
               var _loc12_ = {x:x,y:y};
               localToGlobal(_loc12_);
               _root.game.mazebg.globalToLocal(_loc12_);
               collisionPoints.push(_loc12_);
               _loc13_ = true;
            }
         }
         else if(_loc13_)
         {
            _loc12_ = {x:x,y:y};
            localToGlobal(_loc12_);
            _root.game.mazebg.globalToLocal(_loc12_);
            collisionPoints.push(_loc12_);
            _loc13_ = false;
         }
      }
      if(straightLine)
      {
         while(!hitCheck(_root.game.mazebg,{x:x + 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x - 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x,y:y + 2.5 * (_root.SCALE / 50)}) || !hitCheck(_root.game.mazebg,{x:x,y:y - 2.5 * (_root.SCALE / 50)}))
         {
            x -= xSpeed;
            y -= ySpeed;
            lengthCounter--;
         }
      }
      else
      {
         while(!hitCheck(_root.game.mazebg,{x:x + 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x - 2.5 * (_root.SCALE / 50),y:y}) || !hitCheck(_root.game.mazebg,{x:x,y:y + 2.5 * (_root.SCALE / 50)}) || !hitCheck(_root.game.mazebg,{x:x,y:y - 2.5 * (_root.SCALE / 50)}))
         {
            lengthCounter--;
            x = biggestCircleCenter.x + Math.cos(biggestCircleStartAngle + lengthCounter * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
            y = biggestCircleCenter.y + Math.sin(biggestCircleStartAngle + lengthCounter * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         }
      }
      active = true;
      warmup--;
   }
   if(active)
   {
      ballSize = Math.max(0,ballSize - 5);
      rayThickness = Math.min(5,rayThickness + 1);
      _loc10_ = 0;
      while(_loc10_ < collisionPoints.length)
      {
         var _loc8_ = 0;
         while(_loc8_ < 1)
         {
            if(Math.random() <= 0.5)
            {
               p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
               this.swapDepths(p);
               particleCounter++;
               var _loc15_ = Math.random() * 360;
               _loc14_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
               p.x = collisionPoints[_loc10_].x;
               p.y = collisionPoints[_loc10_].y;
               p._x = p.x;
               p._y = p.y;
               p.lineStyle((Math.random() + 1) * (_root.SCALE / 50),Math.random() <= 0.7000000000000001 ? ownerColor : darkerOwnerColor);
               p.moveTo(0,0);
               var _loc7_ = 0;
               var _loc6_ = 0;
               _loc8_ = 0;
               while(_loc8_ < 8)
               {
                  _loc7_ += Math.random() * 10 - 5;
                  _loc6_ += Math.random() * 10 - 5;
                  p.lineTo(_loc7_ * (_root.SCALE / 50),_loc6_ * (_root.SCALE / 50));
                  _loc8_ = _loc8_ + 1;
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
            _loc8_ = _loc8_ + 1;
         }
         _loc10_ = _loc10_ + 1;
      }
      _loc10_ = 0;
      while(_loc10_ < collisionPoints.length)
      {
         if(Math.random() <= 0.5)
         {
            p = _root.game.mazebg.createEmptyMovieClip("particle" + particleCounter + "-" + _root.game.mazebg.getNextHighestDepth(),_root.game.mazebg.getNextHighestDepth());
            this.swapDepths(p);
            particleCounter++;
            _loc15_ = Math.random() * 360;
            _loc14_ = (0.5 + 2 * Math.random()) * (_root.SCALE / 50);
            p.x = collisionPoints[_loc10_].x;
            p.y = collisionPoints[_loc10_].y;
            p._x = p.x;
            p._y = p.y;
            p.lineStyle((Math.random() * 2 + 1) * (_root.SCALE / 50),5066061);
            p.moveTo(0,0);
            p.lineTo(1,0);
            p.xspeed = Math.cos(_loc15_) * _loc14_;
            p.yspeed = Math.sin(_loc15_) * _loc14_;
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
         _loc10_ = _loc10_ + 1;
      }
      _loc10_ = 0;
      while(_loc10_ < _root.TANKS)
      {
         _loc3_ = _root.game["tank" + _loc10_];
         if(_loc3_ != owner)
         {
            if(_loc3_.alive)
            {
               _loc8_ = 0;
               while(_loc8_ < _loc3_.hitPointsFront.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsFront[_loc8_].x,y:_loc3_.hitPointsFront[_loc8_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc10_);
                     break;
                  }
                  _loc8_ = _loc8_ + 1;
               }
               _loc8_ = 0;
               while(_loc8_ < _loc3_.hitPointsLeft.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsLeft[_loc8_].x,y:_loc3_.hitPointsLeft[_loc8_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc10_);
                     break;
                  }
                  _loc8_ = _loc8_ + 1;
               }
               _loc8_ = 0;
               while(_loc8_ < _loc3_.hitPointsRear.length)
               {
                  if(_loc3_.alive && hitCheck2(_loc3_,{x:_loc3_.hitPointsRear[_loc8_].x,y:_loc3_.hitPointsRear[_loc8_].y}))
                  {
                     _root.registerHit(owner,_loc3_);
                     _root.destroyTank(_loc10_);
                     break;
                  }
                  _loc8_ = _loc8_ + 1;
               }
            }
         }
         _loc10_ = _loc10_ + 1;
      }
   }
   if(!active && warmup < 0)
   {
      rayThickness -= 3;
      if(rayThickness <= 0)
      {
         owner.deathRayReady = true;
         owner.turret.gotoAndStop(20);
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
   _loc10_ = 0;
   while(_loc10_ < fractions.length)
   {
      fractions[_loc10_] -= 5;
      _loc10_ = _loc10_ + 1;
   }
   clear();
   lineStyle(rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation + 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   if(straightLine)
   {
      lineTo(x,y);
   }
   else
   {
      _loc10_ = 0;
      while(_loc10_ <= lengthCounter)
      {
         _loc14_ = (!leftSide ? 1 : -1) * Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
         _loc7_ = biggestCircleCenter.x + Math.cos(biggestCircleStartAngle + _loc10_ * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         _loc6_ = biggestCircleCenter.y + Math.sin(biggestCircleStartAngle + _loc10_ * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         lineTo(_loc7_,_loc6_);
         _loc10_ = _loc10_ + 1;
      }
   }
   lineStyle(0.5 * rayThickness * (_root.SCALE / 50));
   lineGradientStyle("linear",colors,alphas,fractions,{matrixType:"box",x:-750 * (_root.SCALE / 50),y:-750 * (_root.SCALE / 50),w:1500 * (_root.SCALE / 50),h:1500 * (_root.SCALE / 50),r:(owner._rotation - 90) / 180 * 3.141592653589793});
   moveTo(0,0);
   if(straightLine)
   {
      lineTo(x,y);
   }
   else
   {
      _loc10_ = 0;
      while(_loc10_ <= lengthCounter)
      {
         _loc14_ = (!leftSide ? 1 : -1) * Math.sqrt(xSpeed * xSpeed + ySpeed * ySpeed);
         _loc7_ = biggestCircleCenter.x + Math.cos(biggestCircleStartAngle + _loc10_ * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         _loc6_ = biggestCircleCenter.y + Math.sin(biggestCircleStartAngle + _loc10_ * _loc14_ / biggestCircleRadius) * biggestCircleRadius;
         lineTo(_loc7_,_loc6_);
         _loc10_ = _loc10_ + 1;
      }
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
