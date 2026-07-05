function drawHitPoints(points, scale)
{
   i = 0;
   while(i < points.length)
   {
      this.lineStyle(10,65280);
      this.moveTo(points[i].x * scale,points[i].y * scale);
      this.lineTo(points[i].x * scale + 1,points[i].y * scale);
      i++;
   }
}
function hitCheck(points)
{
   i = 0;
   while(i < points.length)
   {
      var _loc2_ = {x:points[i].x,y:points[i].y};
      localToGlobal(_loc2_);
      if(_root.game.mazemc.hitTest(_loc2_.x,_loc2_.y,true))
      {
         return true;
      }
      i++;
   }
   return false;
}
function expandedHitCheck(points, scale)
{
   i = 0;
   while(i < points.length)
   {
      var _loc2_ = {x:points[i].x * scale,y:points[i].y * scale};
      localToGlobal(_loc2_);
      if(_root.game.mazemc.hitTest(_loc2_.x,_loc2_.y,true))
      {
         return true;
      }
      i++;
   }
   return false;
}
forwardSpeed = 4 * (_root.SCALE / 50);
backUpSpeed = 2.5 * (_root.SCALE / 50);
turnSpeed = 10;
triggerReleased = true;
bulletsFired = 0;
laserReady = true;
fragFired = false;
alive = true;
gatlingReady = true;
homingReady = true;
minesLayed = 0;
deathRayReady = true;
remoteControlling = false;
electricReady = true;
x = _X;
y = _Y;
hitPointsFront = new Array();
hitPointsFront[0] = {x:(- base._width) / 2,y:(- base._height) / 2};
hitPointsFront[1] = {x:(- base._width) / 4,y:(- base._height) / 2};
hitPointsFront[2] = {x:base._width / 4,y:(- base._height) / 2};
hitPointsFront[3] = {x:base._width / 2,y:(- base._height) / 2};
hitPointsFront[4] = {x:(- turret._width) / 6,y:(- turret._height) / 16 * 11};
hitPointsFront[5] = {x:turret._width / 6,y:(- turret._height) / 16 * 11};
hitPointsRear = new Array();
hitPointsRear[0] = {x:(- base._width) / 2,y:base._height / 2};
hitPointsRear[1] = {x:(- base._width) / 4,y:base._height / 2};
hitPointsRear[2] = {x:0,y:base._height / 2};
hitPointsRear[3] = {x:base._width / 4,y:base._height / 2};
hitPointsRear[4] = {x:base._width / 2,y:base._height / 2};
hitPointsRight = new Array();
hitPointsRight[0] = {x:base._width / 2,y:(- base._height) / 6 * 2};
hitPointsRight[1] = {x:base._width / 2,y:(- base._height) / 6};
hitPointsRight[2] = {x:base._width / 2,y:0};
hitPointsRight[3] = {x:base._width / 2,y:base._height / 6};
hitPointsRight[4] = {x:base._width / 2,y:base._height / 6 * 2};
hitPointsLeft = new Array();
hitPointsLeft[0] = {x:(- base._width) / 2,y:(- base._height) / 6 * 2};
hitPointsLeft[1] = {x:(- base._width) / 2,y:(- base._height) / 6};
hitPointsLeft[2] = {x:(- base._width) / 2,y:0};
hitPointsLeft[3] = {x:(- base._width) / 2,y:base._height / 6};
hitPointsLeft[4] = {x:(- base._width) / 2,y:base._height / 6 * 2};
if(mouseTank)
{
   this.onMouseDown = function()
   {
      if(!_root.sound.hitTest(_root._xmouse,_root._ymouse,false) && !_root.settings.hitTest(_root._xmouse,_root._ymouse,false))
      {
         fire = true;
      }
   };
   this.onMouseUp = function()
   {
      fire = false;
   };
}
onEnterFrame = function()
{
   var _loc12_ = new Color(this.base.background);
   _loc12_.setTint(this.baseColor.r,this.baseColor.g,this.baseColor.b,this.baseColor.a);
   var _loc11_ = new Color(this.turret.background);
   _loc11_.setTint(this.turretColor.r,this.turretColor.g,this.turretColor.b,this.turretColor.a);
   _loc12_ = new Color(this.scoreboard.tankIcon.tracks.background);
   _loc12_.setTint(this.baseColor.r,this.baseColor.g,this.baseColor.b,this.baseColor.a);
   _loc11_ = new Color(this.scoreboard.tankIcon.turretBackground);
   _loc11_.setTint(this.turretColor.r,this.turretColor.g,this.turretColor.b,this.turretColor.a);
   if(mouseTank)
   {
      var _loc9_ = {x:_root.xMouse,y:_root.yMouse};
      _root.game.mazemc.globalToLocal(_loc9_);
      deltaX = _loc9_.x - _X;
      deltaY = _loc9_.y - _Y;
      deltaLength = Math.sqrt(Math.pow(deltaX,2) + Math.pow(deltaY,2));
      _root.scopeCross._x = _root.xMouse;
      _root.scopeCross._y = _root.yMouse;
      if(deltaLength > 60 && alive)
      {
         _root.scopeCircle._x = _root.game._x + _X + deltaX / deltaLength * 60;
         _root.scopeCircle._y = _root.game._y + _Y + deltaY / deltaLength * 60;
      }
      else
      {
         _root.scopeCircle._x = _root.xMouse;
         _root.scopeCircle._y = _root.yMouse;
      }
      if(_root.settingsUseNewMouseControl)
      {
         if(deltaLength > 120 && alive)
         {
            _root.scopeCircle.gotoAndStop(2);
         }
         else
         {
            _root.scopeCircle.gotoAndStop(1);
         }
      }
   }
   if(_root.frozen)
   {
      return undefined;
   }
   if(alive && !_root.lockedControl(this,currentWeapon))
   {
      oldX = x;
      oldY = y;
      oldRot = _rotation;
      if(mouseTank)
      {
         if(_root.settingsUseNewMouseControl)
         {
            var _loc7_ = _rotation;
            var _loc6_ = undefined;
            var _loc8_ = false;
            var _loc10_ = deltaLength < 120;
            if(deltaX != 0)
            {
               if(deltaX > 0)
               {
                  _loc6_ = 90 + Math.atan(deltaY / deltaX) * 180 / 3.141592653589793;
               }
               else
               {
                  _loc6_ = -90 + Math.atan(deltaY / deltaX) * 180 / 3.141592653589793;
               }
            }
            else if(deltaY > 0)
            {
               _loc6_ = 180;
            }
            else if(deltaY < 0)
            {
               _loc6_ = 0;
            }
            else
            {
               _loc6_ = _loc7_;
            }
            _loc6_ = Math.round(_loc6_ / turnSpeed) * turnSpeed;
            if(_loc10_ && Math.abs(_loc6_ - _loc7_) > 90 && Math.abs(_loc6_ - _loc7_) < 270)
            {
               _loc8_ = true;
               _loc6_ += 180;
               if(_loc6_ > 180)
               {
                  _loc6_ -= 360;
               }
            }
            if(_loc6_ > _loc7_)
            {
               if(Math.abs(_loc6_ - _loc7_) > 180)
               {
                  turnLeft = true;
                  turnRight = false;
               }
               else
               {
                  turnLeft = false;
                  turnRight = true;
               }
            }
            else if(_loc6_ < _loc7_)
            {
               if(Math.abs(_loc6_ - _loc7_) > 180)
               {
                  turnLeft = false;
                  turnRight = true;
               }
               else
               {
                  turnLeft = true;
                  turnRight = false;
               }
            }
            else
            {
               turnLeft = false;
               turnRight = false;
            }
         }
         else
         {
            if(deltaX < 0)
            {
               if(deltaY < 0)
               {
                  aimAngle = -3.141592653589793 + Math.atan(deltaY / deltaX);
               }
               else
               {
                  aimAngle = 3.141592653589793 + Math.atan(deltaY / deltaX);
               }
            }
            else if(deltaX > 0)
            {
               aimAngle = Math.atan(deltaY / deltaX);
            }
            else if(deltaY < 0)
            {
               aimAngle = -1.5707963267948966;
            }
            else
            {
               aimAngle = 1.5707963267948966;
            }
            _rotation = (aimAngle + 1.5707963267948966) * 180 / 3.141592653589793;
         }
         if(_root.settingsUseNewMouseControl)
         {
            if(deltaLength > 60 && (Math.abs(_loc6_ - _loc7_) < 45 || Math.abs(_loc6_ - _loc7_) > 315))
            {
               forward = !_loc8_;
               backup = _loc8_;
            }
            else
            {
               forward = false;
               backup = false;
            }
         }
         else if(deltaLength > 60)
         {
            forward = true;
            backup = false;
         }
         else
         {
            forward = false;
            backup = false;
         }
      }
      else
      {
         if(Key.isDown(KEYTURNLEFT))
         {
            turnLeft = true;
         }
         else
         {
            turnLeft = false;
         }
         if(Key.isDown(KEYFORWARD))
         {
            forward = true;
         }
         else
         {
            forward = false;
         }
         if(Key.isDown(KEYTURNRIGHT))
         {
            turnRight = true;
         }
         else
         {
            turnRight = false;
         }
         if(Key.isDown(KEYBACKUP))
         {
            backup = true;
         }
         else
         {
            backup = false;
         }
         if(Key.isDown(KEYFIRE))
         {
            fire = true;
         }
         else
         {
            fire = false;
         }
      }
      if(AI != undefined)
      {
         if(AI.makeDecisionsAndUpdateGoal())
         {
            AI.decideActionsToAchieveGoal();
         }
         AI.setInputToDoActions();
      }
      STEPS = 5;
      movesSuc = 0;
      turnsSuc = 0;
      moveSize = 0;
      turnSize = 0;
      if(forward)
      {
         moveSize = forwardSpeed / STEPS;
      }
      if(backup)
      {
         moveSize -= backUpSpeed / STEPS;
      }
      if(turnLeft)
      {
         turnSize = (- turnSpeed) / STEPS;
      }
      if(turnRight)
      {
         turnSize += turnSpeed / STEPS;
      }
      hitSomething = false;
      var _loc5_ = 0;
      while(_loc5_ < STEPS)
      {
         _rotation = _rotation + turnSize;
         x += Math.cos((_rotation - 90) * 3.141592653589793 / 180) * moveSize;
         y += Math.sin((_rotation - 90) * 3.141592653589793 / 180) * moveSize;
         _loc5_ = _loc5_ + 1;
      }
      _X = x;
      _Y = y;
      if(hitCheck(hitPointsFront) || hitCheck(hitPointsRear) || hitCheck(hitPointsLeft) || hitCheck(hitPointsRight))
      {
         x = oldX;
         y = oldY;
         _X = oldX;
         _Y = oldY;
         _rotation = oldRot;
         _loc5_ = 0;
         while(_loc5_ < STEPS)
         {
            if(!mouseTank)
            {
               oldRot = _rotation;
               _rotation = _rotation + turnSize;
            }
            if(hitCheck(hitPointsFront) || hitCheck(hitPointsRear) || hitCheck(hitPointsLeft) || hitCheck(hitPointsRight))
            {
               _rotation = oldRot;
               hitSomething = true;
            }
            oldX = x;
            oldY = y;
            x += Math.cos((_rotation - 90) * 3.141592653589793 / 180) * moveSize;
            y += Math.sin((_rotation - 90) * 3.141592653589793 / 180) * moveSize;
            _X = x;
            _Y = y;
            if(moveSize > 0 && hitCheck(hitPointsFront))
            {
               x = oldX;
               y = oldY;
               _X = oldX;
               _Y = oldY;
               hitSomething = true;
            }
            else if(moveSize < 0 && hitCheck(hitPointsRear))
            {
               x = oldX;
               y = oldY;
               _X = oldX;
               _Y = oldY;
               hitSomething = true;
            }
            _loc5_ = _loc5_ + 1;
         }
      }
      offset = (360 + _rotation) % turnSpeed;
      if(!hitSomething && turnSize != 0 && offset != 0)
      {
         if(offset < turnSpeed / 2)
         {
            _rotation = _rotation - offset;
         }
         else
         {
            _rotation = _rotation + (turnSpeed - offset);
         }
      }
      if(fire && triggerReleased && _root.weaponReady(this,currentWeapon))
      {
         triggerReleased = false;
         _root.fireWeapon(this,currentWeapon);
      }
      else if(!fire)
      {
         triggerReleased = true;
      }
   }
   if(equipment != undefined)
   {
      if(currentEquipment == "shield")
      {
         if(equipment.getDepth() < getDepth())
         {
            equipment.swapDepths(_root.game.getNextHighestDepth());
         }
         equipment._x = _X;
         equipment._y = _Y;
      }
   }
};
