function updateGoal(temp)
{
   if(myGoal.priority < temp.priority)
   {
      myGoal = temp;
   }
}
function dodgeTrajectories(fieldx, fieldy, bullets, maxTimeToDodge, maxDistToDodge, maxCellDistToDodge, hitCheckInterval, checkBounce)
{
   var _loc39_ = maxTimeToDodge;
   var _loc8_ = maxDistToDodge;
   var _loc23_ = {priority:0};
   var _loc18_ = 0;
   while(_loc18_ < bullets.length)
   {
      var _loc2_ = bullets[_loc18_];
      var _loc4_ = _loc2_.x;
      var _loc3_ = _loc2_.y;
      var _loc20_ = Math.floor(_loc4_ / _root.SCALE);
      var _loc21_ = Math.floor(_loc3_ / _root.SCALE);
      if(_root.distancesForMaze[fieldx][fieldy][_loc20_][_loc21_] <= maxCellDistToDodge)
      {
         var _loc10_ = _loc2_.x + _loc2_.xSpeed * hitCheckInterval;
         var _loc9_ = _loc2_.y + _loc2_.ySpeed * hitCheckInterval;
         var _loc11_ = myTank.x;
         var _loc15_ = myTank.y;
         var _loc22_ = (_loc10_ - _loc4_) * (_loc10_ - _loc4_) + (_loc9_ - _loc3_) * (_loc9_ - _loc3_);
         var _loc26_ = ((_loc11_ - _loc4_) * (_loc10_ - _loc4_) + (_loc15_ - _loc3_) * (_loc9_ - _loc3_)) / _loc22_;
         if(_loc26_ > -1 && _loc26_ < _loc39_)
         {
            var _loc14_ = _loc4_ + _loc26_ * (_loc10_ - _loc4_);
            var _loc13_ = _loc3_ + _loc26_ * (_loc9_ - _loc3_);
            var _loc7_ = _loc11_ - _loc14_;
            var _loc6_ = _loc15_ - _loc13_;
            var _loc36_ = Math.sqrt(_loc7_ * _loc7_ + _loc6_ * _loc6_);
            var _loc17_ = checkPathForCollision(_loc14_,_loc13_,_loc7_ / _loc36_,_loc6_ / _loc36_,1,Math.ceil(_loc36_),Math.ceil(_loc36_));
            if(_loc17_ == undefined && _loc36_ < _loc8_)
            {
               _loc7_ = _loc10_ - _loc14_;
               _loc6_ = _loc9_ - _loc13_;
               var _loc12_ = Math.sqrt(_loc7_ * _loc7_ + _loc6_ * _loc6_);
               _loc17_ = checkPathForCollision(_loc14_,_loc13_,_loc7_ / _loc12_,_loc6_ / _loc12_,1,Math.ceil(_loc12_),Math.ceil(_loc12_));
               if(_loc17_ == undefined)
               {
                  _loc8_ = Math.min(_loc8_,_loc36_);
                  _loc23_ = {goal:"dodgeBullet",x:_loc2_.x,y:_loc2_.y,closest:{x:_loc14_,y:_loc13_},dist:_loc36_,t:_loc26_,dir:{x:_loc10_ - _loc4_,y:_loc9_ - _loc3_},maxTime:maxTimeToDodge,maxDist:maxDistToDodge,period:10,priority:1,updateContinuously:false,id:goalId++};
               }
            }
         }
         if(_loc8_ > _root.SCALE / 4 && checkBounce)
         {
            var _loc5_ = checkPathForCollision(_loc4_,_loc3_,_loc2_.xSpeed,_loc2_.ySpeed,hitCheckInterval,12,_loc2_.lifetime);
            if(_loc5_ != undefined)
            {
               _loc4_ = _loc5_.x;
               _loc3_ = _loc5_.y;
               _loc10_ = _loc5_.x + _loc5_.xSpeed * hitCheckInterval;
               _loc9_ = _loc5_.y + _loc5_.ySpeed * hitCheckInterval;
               _loc22_ = (_loc10_ - _loc4_) * (_loc10_ - _loc4_) + (_loc9_ - _loc3_) * (_loc9_ - _loc3_);
               _loc26_ = ((_loc11_ - _loc4_) * (_loc10_ - _loc4_) + (_loc15_ - _loc3_) * (_loc9_ - _loc3_)) / _loc22_;
               if(_loc26_ > 0 && _loc26_ < maxTimeToDodge - _loc5_.t)
               {
                  _loc14_ = _loc4_ + _loc26_ * (_loc10_ - _loc4_);
                  _loc13_ = _loc3_ + _loc26_ * (_loc9_ - _loc3_);
                  _loc7_ = _loc11_ - _loc14_;
                  _loc6_ = _loc15_ - _loc13_;
                  _loc36_ = Math.sqrt(_loc7_ * _loc7_ + _loc6_ * _loc6_);
                  _loc17_ = checkPathForCollision(_loc14_,_loc13_,_loc7_ / _loc36_,_loc6_ / _loc36_,1,Math.ceil(_loc36_),Math.ceil(_loc36_));
                  if(_loc17_ == undefined && _loc36_ < _loc8_)
                  {
                     _loc7_ = _loc14_ - _loc4_;
                     _loc6_ = _loc13_ - _loc3_;
                     _loc12_ = Math.sqrt(_loc7_ * _loc7_ + _loc6_ * _loc6_);
                     _loc17_ = checkPathForCollision(_loc4_,_loc3_,_loc7_ / _loc12_,_loc6_ / _loc12_,1,Math.ceil(_loc12_),Math.ceil(_loc12_));
                     if(_loc17_ == undefined)
                     {
                        _loc8_ = Math.min(_loc8_,_loc36_);
                        _loc23_ = {goal:"dodgeBullet",x:_loc2_.x,y:_loc2_.y,closest:{x:_loc14_,y:_loc13_},dist:_loc36_,t:_loc26_ + _loc5_.t,dir:{x:_loc10_ - _loc4_,y:_loc9_ - _loc3_},maxTime:maxTimeToDodge,maxDist:maxDistToDodge,period:10,priority:1,updateContinuously:false,id:goalId++};
                     }
                  }
               }
            }
         }
      }
      _loc18_ = _loc18_ + 1;
   }
   return _loc23_;
}
function tryToRetaliate()
{
   if(currentAggresiveness < AGGRESIVENESS / 2)
   {
      return undefined;
   }
   switch(myTank.currentWeapon)
   {
      case "bullet":
      case "laser":
         if(myTank.bulletsFired < _root.settingsMaxBullets)
         {
            var _loc14_ = myTank._rotation;
            var _loc12_ = false;
            var _loc13_ = _root.BULLETLIFETIME;
            var _loc11_ = _root.MOVIEWIDTH + _root.MOVIEHEIGHT;
            var _loc10_ = checkBulletPath(_loc14_);
            if(_loc10_.result == "HIT")
            {
               _loc12_ = true;
               if(_loc10_.time < _loc13_)
               {
                  _loc13_ = _loc10_.time;
                  _loc11_ = 0;
               }
            }
            else if(_loc10_.result == "NOTHING" && !_loc12_)
            {
               if(_loc10_.closest < _loc11_)
               {
                  _loc11_ = _loc10_.closest;
               }
            }
            if(_loc12_ || _loc11_ < MAXCLOSESTDISTANCE / 2)
            {
               trace("Retaliate!");
               myActionsForGoal.push({action:"fireWeapon",delay:1});
               currentAggresiveness = Math.max(0,currentAggresiveness - 0.2);
            }
         }
         break;
      case "frag":
         if(myTank.fragFired)
         {
            var _loc4_ = myTank.lastFrag;
            var _loc6_ = myTank.x - _loc4_.x;
            var _loc5_ = myTank.y - _loc4_.y;
            var _loc3_ = Math.sqrt(_loc6_ * _loc6_ + _loc5_ * _loc5_);
            var _loc7_ = checkPathForCollision(_loc4_.x,_loc4_.y,_loc6_ / _loc3_,_loc5_ / _loc3_,1,Math.ceil(_loc3_),Math.ceil(_loc3_));
            if(_loc7_ != undefined || _loc3_ >= FRAGBOMBSAFETYDIST)
            {
               var _loc2_ = 0;
               while(_loc2_ < _root.TANKS)
               {
                  if(_root.game["tank" + _loc2_].alive && _root.game["tank" + _loc2_] != myTank)
                  {
                     _loc6_ = _root.game["tank" + _loc2_].x - _loc4_.x;
                     _loc5_ = _root.game["tank" + _loc2_].y - _loc4_.y;
                     _loc3_ = Math.sqrt(_loc6_ * _loc6_ + _loc5_ * _loc5_);
                     if(_loc3_ <= FRAGBOMBDETONATEDIST)
                     {
                        _loc7_ = checkPathForCollision(_loc4_.x,_loc4_.y,_loc6_ / _loc3_,_loc5_ / _loc3_,1,Math.ceil(_loc3_),Math.ceil(_loc3_));
                        if(_loc7_ == undefined)
                        {
                           myActionsForGoal.push({action:"fireWeapon",delay:1});
                        }
                     }
                  }
                  _loc2_ = _loc2_ + 1;
               }
            }
         }
         break;
      case "gatling":
   }
}
function checkPathForCollision(x, y, xSpeed, ySpeed, hitCheckInterval, maxtime, lifetime)
{
   lifetime = Math.min(maxtime,lifetime);
   t = 0;
   while(lifetime > 0)
   {
      i = 0;
      while(i < hitCheckInterval)
      {
         previousX = x;
         previousY = y;
         x += xSpeed;
         y += ySpeed;
         if(_root.game.mazemc.hitTest(_root.game._x + x,_root.game._y + y,true))
         {
            x = previousX;
            y = previousY;
            x -= xSpeed;
            y += ySpeed;
            if(_root.game.mazemc.hitTest(_root.game._x + x,_root.game._y + y,true))
            {
               hitOnXInvert = true;
            }
            else
            {
               hitOnXInvert = false;
            }
            x = previousX;
            y = previousY;
            x += xSpeed;
            y -= ySpeed;
            if(_root.game.mazemc.hitTest(_root.game._x + x,_root.game._y + y,true))
            {
               hitOnYInvert = true;
            }
            else
            {
               hitOnYInvert = false;
            }
            if(hitOnXInvert && !hitOnYInvert)
            {
               ySpeed = - ySpeed;
            }
            else if(hitOnYInvert && !hitOnXInvert)
            {
               xSpeed = - xSpeed;
            }
            else
            {
               xSpeed = - xSpeed;
               ySpeed = - ySpeed;
            }
            x = previousX;
            y = previousY;
            x += xSpeed;
            y += ySpeed;
            return {x:x,y:y,xSpeed:xSpeed,ySpeed:ySpeed,t:t};
         }
         i++;
      }
      lifetime = lifetime - 1;
      t++;
   }
   return undefined;
}
function checkBulletPath(angle)
{
   var _loc3_ = myTank._x + Math.cos((angle - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
   var _loc2_ = myTank._y + Math.sin((angle - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
   var _loc5_ = Math.cos((angle - 90) * 3.141592653589793 / 180) * _root.BULLETSPEED * (_root.SCALE / 50);
   var _loc4_ = Math.sin((angle - 90) * 3.141592653589793 / 180) * _root.BULLETSPEED * (_root.SCALE / 50);
   var _loc7_ = _root.BULLETLIFETIME / 3;
   var _loc11_ = _root.BULLETDEADLY;
   var _loc10_ = _root.MOVIEWIDTH + _root.MOVIEHEIGHT;
   while(_loc7_ > 0)
   {
      i = 0;
      while(i < 1)
      {
         previousX = _loc3_;
         previousY = _loc2_;
         _loc3_ += _loc5_;
         _loc2_ += _loc4_;
         if(_root.game.mazemc.hitTest(_root.game._x + _loc3_,_root.game._y + _loc2_,true))
         {
            _loc3_ = previousX;
            _loc2_ = previousY;
            _loc3_ -= _loc5_;
            _loc2_ += _loc4_;
            if(_root.game.mazemc.hitTest(_root.game._x + _loc3_,_root.game._y + _loc2_,true))
            {
               hitOnXInvert = true;
            }
            else
            {
               hitOnXInvert = false;
            }
            _loc3_ = previousX;
            _loc2_ = previousY;
            _loc3_ += _loc5_;
            _loc2_ -= _loc4_;
            if(_root.game.mazemc.hitTest(_root.game._x + _loc3_,_root.game._y + _loc2_,true))
            {
               hitOnYInvert = true;
            }
            else
            {
               hitOnYInvert = false;
            }
            if(hitOnXInvert && !hitOnYInvert)
            {
               _loc4_ = - _loc4_;
            }
            else if(hitOnYInvert && !hitOnXInvert)
            {
               _loc5_ = - _loc5_;
            }
            else
            {
               _loc5_ = - _loc5_;
               _loc4_ = - _loc4_;
            }
            _loc3_ = previousX;
            _loc2_ = previousY;
            _loc3_ += _loc5_;
            _loc2_ += _loc4_;
         }
         i++;
      }
      if(_loc11_ == 0)
      {
         var i = 0;
         while(i < _root.TANKS)
         {
            if(_root.game["tank" + i].alive && _root.game["tank" + i].hitTest(_root.game._x + _loc3_,_root.game._y + _loc2_,false))
            {
               if(_root.game["tank" + i].hitTest(_root.game._x + _loc3_,_root.game._y + _loc2_,true))
               {
                  if(_root.game["tank" + i] == myTank)
                  {
                     return {result:"SUICIDE",time:_root.BULLETLIFETIME / 3 - _loc7_};
                  }
                  return {result:"HIT",time:_root.BULLETLIFETIME / 3 - _loc7_};
               }
            }
            else if(_root.game["tank" + i].alive && _root.game["tank" + i] != myTank)
            {
               var _loc6_ = Math.abs(_root.game["tank" + i].x - _loc3_) + Math.abs(_root.game["tank" + i].y - _loc2_);
               if(_loc6_ < MAXCLOSESTDISTANCE)
               {
                  var _loc8_ = Math.floor(_loc3_ / _root.SCALE);
                  var _loc9_ = Math.floor(_loc2_ / _root.SCALE);
                  if(_root.distancesForMaze[_root.tankFields[i].x][_root.tankFields[i].y][_loc8_][_loc9_] <= MAXCLOSESTCELLDISTANCE)
                  {
                     if(_loc6_ < _loc10_)
                     {
                        _loc10_ = _loc6_;
                     }
                  }
               }
            }
            i++;
         }
      }
      if(_loc11_ > 0)
      {
         _loc11_ = _loc11_ - 1;
      }
      _loc7_ = _loc7_ - 1;
   }
   return {result:"NOTHING",time:_root.BULLETLIFETIME / 3,closest:_loc10_};
}
function pushActionsToFollowPath(path)
{
   var _loc2_ = path.length - 1;
   while(_loc2_ >= 1)
   {
      myActionsForGoal.push({action:"driveToField",x:path[_loc2_].x,y:path[_loc2_].y});
      _loc2_ = _loc2_ - 1;
   }
   if(path.length > 1)
   {
      var _loc6_ = myTank._rotation;
      var _loc5_ = undefined;
      var _loc4_ = {x:(path[1].x + 0.5) * _root.SCALE - myTank._x,y:(path[1].y + 0.5) * _root.SCALE - myTank._y};
      if(_loc4_.x != 0)
      {
         if(_loc4_.x > 0)
         {
            _loc5_ = 90 + Math.atan(_loc4_.y / _loc4_.x) * 180 / 3.141592653589793;
         }
         else
         {
            _loc5_ = -90 + Math.atan(_loc4_.y / _loc4_.x) * 180 / 3.141592653589793;
         }
      }
      else if(_loc4_.y > 0)
      {
         _loc5_ = 180;
      }
      else if(_loc4_.y < 0)
      {
         _loc5_ = 0;
      }
      else
      {
         _loc5_ = _loc6_;
      }
   }
   myActionsForGoal.push({action:"driveToPos",x:(path[0].x + 0.5) * _root.SCALE,y:(path[0].y + 0.5) * _root.SCALE,canReverse:path.length <= 2});
}
function makeDecisionsAndUpdateGoal()
{
   if(myGoal.period > 0)
   {
      myGoal.period--;
      return myGoal.updateContinuously;
   }
   myGoal.priority *= 0.9000000000000002;
   oldGoal = myGoal;
   var _loc6_ = Math.floor(myTank._x / _root.SCALE);
   var _loc7_ = Math.floor(myTank._y / _root.SCALE);
   if(_root.aliveCount > 1 && myTank.currentWeapon == "bullet")
   {
      var _loc25_ = new Array();
      for(var _loc44_ in _root.game.mazebg)
      {
         if(_loc44_.substr(0,5) == "crate")
         {
            _loc25_.push(_root.game.mazebg[_loc44_]);
         }
      }
      var _loc37_ = MAXCELLDISTTOGOFORCRATE;
      var _loc46_ = {priority:0};
      var _loc8_ = 0;
      while(_loc8_ < _loc25_.length)
      {
         var _loc23_ = _loc25_[_loc8_];
         var _loc22_ = Math.floor(_loc23_._x / _root.SCALE);
         var _loc21_ = Math.floor(_loc23_._y / _root.SCALE);
         var _loc53_ = _root.distancesForMaze[_loc6_][_loc7_][_loc22_][_loc21_];
         if(_loc53_ <= _loc37_)
         {
            _loc37_ = _loc53_;
            _loc46_ = {goal:"goForCrate",x:_loc22_,y:_loc21_,period:10,priority:(MAXCELLDISTTOGOFORCRATE - _loc53_) / MAXCELLDISTTOGOFORCRATE * GREEDY * (_root.settingsMaxBullets - myTank.bulletsFired) / _root.settingsMaxBullets,updateContinuously:false,id:goalId++};
         }
         _loc8_ = _loc8_ + 1;
      }
      updateGoal(_loc46_);
   }
   var _loc41_ = new Array();
   for(var _loc45_ in _root.game)
   {
      if(_loc45_.substr(0,6) == "bullet")
      {
         _loc41_.push(_root.game[_loc45_]);
      }
   }
   _loc46_ = dodgeTrajectories(_loc6_,_loc7_,_loc41_,MAXTIMETODODGEBULLET,MAXDISTTODODGEBULLET,MAXCELLDISTTODODGEBULLET,_root.BULLETHITCHECKINTERVALS,true);
   updateGoal(_loc46_);
   var _loc26_ = new Array();
   var _loc42_ = new Array();
   for(var _loc43_ in _root.game)
   {
      if(_loc43_.substr(0,4) == "frag")
      {
         if(_loc43_.substr(0,12) == "fragfragment")
         {
            if(_root.game[_loc43_].active)
            {
               _loc42_.push(_root.game[_loc43_]);
            }
         }
         else
         {
            _loc26_.push(_root.game[_loc43_]);
         }
      }
   }
   var _loc15_ = MAXCELLDISTTODODGEFRAGBOMB;
   _loc8_ = 0;
   while(_loc8_ < _loc26_.length)
   {
      var _loc17_ = _loc26_[_loc8_];
      var _loc32_ = Math.floor(_loc17_.x / _root.SCALE);
      var _loc30_ = Math.floor(_loc17_.y / _root.SCALE);
      _loc53_ = _root.distancesForMaze[_loc6_][_loc7_][_loc32_][_loc30_];
      if(_loc53_ < _loc15_)
      {
         _loc15_ = _loc53_;
         _loc46_ = {goal:"dodgeFragbomb",frag:_loc17_,period:10,priority:1,updateContinuously:false,id:goalId++};
      }
      _loc8_ = _loc8_ + 1;
   }
   updateGoal(_loc46_);
   _loc46_ = dodgeTrajectories(_loc6_,_loc7_,_loc42_,MAXTIMETODODGEFRAGBOMBFRAGMENT,MAXDISTTODODGEFRAGBOMBFRAGMENT,MAXCELLDISTTODODGEFRAGBOMBFRAGMENT,_root.FRAGHITCHECKINTERVALS,false);
   updateGoal(_loc46_);
   _loc41_ = new Array();
   for(_loc45_ in _root.game)
   {
      if(_loc45_.substr(0,13) == "gatlingBullet")
      {
         _loc41_.push(_root.game[_loc45_]);
      }
   }
   _loc46_ = dodgeTrajectories(_loc6_,_loc7_,_loc41_,MAXTIMETODODGEGATLINGBULLET,MAXDISTTODODGEGATLINGBULLET,MAXCELLDISTTODODGEGATLINGBULLET,_root.GATLINGHITCHECKINTERVALS,true);
   updateGoal(_loc46_);
   _loc15_ = MAXCELLDISTTODODGELASER;
   _loc46_ = {priority:0};
   _loc8_ = 0;
   while(_loc8_ < _root.TANKS)
   {
      if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_].currentEquipment == "aimer" && _root.game["tank" + _loc8_] != myTank)
      {
         var _loc2_ = _root.game["tank" + _loc8_].equipment;
         if(_loc2_.hit == myTank)
         {
            _loc15_ = 0;
            _loc46_ = {goal:"dodgeLaser",dir:{x:_loc2_.hitXSpeed,y:_loc2_.hitYSpeed},owner:_root.game["tank" + _loc8_],period:10,priority:1,updateContinuously:false,id:goalId++};
         }
         else if(_loc2_.hit == undefined)
         {
            var _loc29_ = Math.floor((_loc2_._x + _loc2_.x) / _root.SCALE);
            var _loc31_ = Math.floor((_loc2_._y + _loc2_.y) / _root.SCALE);
            _loc53_ = _root.distancesForMaze[_loc6_][_loc7_][_loc29_][_loc31_];
            if(_loc53_ <= _loc15_)
            {
               var _loc11_ = checkPathForCollision(_loc2_._x + _loc2_.x,_loc2_._y + _loc2_.y,_loc2_.xSpeed,_loc2_.ySpeed,_root.AIMERHITCHECKINTERVALS,12,12);
               if(_loc11_ != undefined)
               {
                  var _loc10_ = _loc2_._x + _loc2_.x;
                  var _loc9_ = _loc2_._y + _loc2_.y;
                  var _loc13_ = _loc11_.x;
                  var _loc12_ = _loc11_.y;
                  var _loc27_ = myTank.x;
                  var _loc35_ = myTank.y;
                  var _loc34_ = (_loc13_ - _loc10_) * (_loc13_ - _loc10_) + (_loc12_ - _loc9_) * (_loc12_ - _loc9_);
                  var _loc14_ = ((_loc27_ - _loc10_) * (_loc13_ - _loc10_) + (_loc35_ - _loc9_) * (_loc12_ - _loc9_)) / _loc34_;
                  if(_loc14_ > 0 && _loc14_ < 1)
                  {
                     var _loc28_ = Math.floor((_loc10_ + _loc14_ * (_loc13_ - _loc10_)) / _root.SCALE);
                     var _loc36_ = Math.floor((_loc9_ + _loc14_ * (_loc12_ - _loc9_)) / _root.SCALE);
                     _loc53_ = _root.distancesForMaze[_loc6_][_loc7_][_loc28_][_loc36_];
                     if(_loc53_ <= _loc15_)
                     {
                        _loc15_ = _loc53_;
                        _loc46_ = {goal:"dodgeLaser",dir:{x:_loc11_.xSpeed,y:_loc11_.ySpeed},owner:_root.game["tank" + _loc8_],period:10,priority:1,updateContinuously:false,id:goalId++};
                     }
                  }
               }
               else
               {
                  _loc15_ = _loc53_;
                  _loc46_ = {goal:"dodgeLaser",dir:{x:_loc2_.xSpeed,y:_loc2_.ySpeed},owner:_root.game["tank" + _loc8_],period:10,priority:1,updateContinuously:false,id:goalId++};
               }
            }
         }
      }
      _loc8_ = _loc8_ + 1;
   }
   updateGoal(_loc46_);
   switch(myTank.currentWeapon)
   {
      case "bullet":
      case "laser":
         if(myTank.bulletsFired < _root.settingsMaxBullets || myTank.currentWeapon == "laser")
         {
            _loc8_ = 0;
            while(_loc8_ < _root.TANKS)
            {
               if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_] != myTank)
               {
                  var _loc16_ = _root.getShortestPathWithDistances(_root.maze,_root.distancesForMaze[_loc6_][_loc7_],_loc6_,_loc7_,_root.tankFields[_loc8_].x,_root.tankFields[_loc8_].y);
                  if(_loc16_.length < LONGESTPATHTOSHOOT)
                  {
                     _loc46_ = {goal:"shootAfter",target:_root.game["tank" + _loc8_],period:10,priority:(_loc16_.length > LONGESTPATHTONOTHESITATETOSHOOT ? (LONGESTPATHTOSHOOT - _loc16_.length) / LONGESTPATHTOSHOOT * currentAggresiveness : 1),updateContinuously:false,id:goalId++};
                     updateGoal(_loc46_);
                  }
               }
               _loc8_ = _loc8_ + 1;
            }
         }
         break;
      case "frag":
         if(!myTank.fragFired)
         {
            _loc8_ = 0;
            while(_loc8_ < _root.TANKS)
            {
               if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_] != myTank)
               {
                  _loc16_ = _root.getShortestPathWithDistances(_root.maze,_root.distancesForMaze[_loc6_][_loc7_],_loc6_,_loc7_,_root.tankFields[_loc8_].x,_root.tankFields[_loc8_].y);
                  if(_loc16_.length < LONGESTPATHTOSHOOT)
                  {
                     _loc46_ = {goal:"shootAfter",target:_root.game["tank" + _loc8_],period:10,priority:(_loc16_.length > LONGESTPATHTONOTHESITATETOSHOOT ? (LONGESTPATHTOSHOOT - _loc16_.length) / LONGESTPATHTOSHOOT * currentAggresiveness : 1),updateContinuously:false,id:goalId++};
                     updateGoal(_loc46_);
                  }
               }
               _loc8_ = _loc8_ + 1;
            }
         }
         else
         {
            _loc17_ = myTank.lastFrag;
            var _loc19_ = myTank.x - _loc17_.x;
            var _loc18_ = myTank.y - _loc17_.y;
            _loc53_ = Math.sqrt(_loc19_ * _loc19_ + _loc18_ * _loc18_);
            var _loc33_ = checkPathForCollision(_loc17_.x,_loc17_.y,_loc19_ / _loc53_,_loc18_ / _loc53_,1,Math.ceil(_loc53_),Math.ceil(_loc53_));
            if(_loc33_ != undefined || _loc53_ >= FRAGBOMBSAFETYDIST)
            {
               _loc8_ = 0;
               while(_loc8_ < _root.TANKS)
               {
                  if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_] != myTank)
                  {
                     _loc19_ = _root.game["tank" + _loc8_].x - _loc17_.x;
                     _loc18_ = _root.game["tank" + _loc8_].y - _loc17_.y;
                     _loc53_ = Math.sqrt(_loc19_ * _loc19_ + _loc18_ * _loc18_);
                     if(_loc53_ <= FRAGBOMBDETONATEDIST)
                     {
                        _loc33_ = checkPathForCollision(_loc17_.x,_loc17_.y,_loc19_ / _loc53_,_loc18_ / _loc53_,1,Math.ceil(_loc53_),Math.ceil(_loc53_));
                        if(_loc33_ == undefined)
                        {
                           _loc46_ = {goal:"detonate",period:1,priority:1,updateContiuously:false,id:goalId++};
                           updateGoal(_loc46_);
                        }
                     }
                  }
                  _loc8_ = _loc8_ + 1;
               }
            }
         }
         break;
      case "gatling":
         if(myTank.gatlingReady)
         {
            _loc8_ = 0;
            while(_loc8_ < _root.TANKS)
            {
               if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_] != myTank)
               {
                  _loc16_ = _root.getShortestPathWithDistances(_root.maze,_root.distancesForMaze[_loc6_][_loc7_],_loc6_,_loc7_,_root.tankFields[_loc8_].x,_root.tankFields[_loc8_].y);
                  if(_loc16_.length < LONGESTPATHTOSHOOT)
                  {
                     _loc46_ = {goal:"sprayBullets",target:_root.game["tank" + _loc8_],period:15,priority:(_loc16_.length > LONGESTPATHTONOTHESITATETOSHOOT ? (LONGESTPATHTOSHOOT - _loc16_.length) / LONGESTPATHTOSHOOT * currentAggresiveness : 1),updateContinuously:false,id:goalId++};
                     updateGoal(_loc46_);
                  }
               }
               _loc8_ = _loc8_ + 1;
            }
         }
   }
   if(_root.aliveCount > 1 && myTank.currentWeapon == "bullet" && myTank.bulletsFired == _root.settingsMaxBullets)
   {
      var _loc4_ = new Array(_root.maze.length - 1);
      _loc8_ = 0;
      while(_loc8_ < _loc4_.length)
      {
         _loc4_[_loc8_] = new Array(_root.maze[_loc8_].length - 1);
         _loc8_ = _loc8_ + 1;
      }
      var _loc5_ = 0;
      while(_loc5_ < _loc4_.length)
      {
         var _loc3_ = 0;
         while(_loc3_ < _loc4_[0].length)
         {
            _loc4_[_loc5_][_loc3_] = 0;
            _loc3_ = _loc3_ + 1;
         }
         _loc5_ = _loc5_ + 1;
      }
      _loc8_ = 0;
      while(_loc8_ < _root.TANKS)
      {
         if(_root.game["tank" + _loc8_].alive && _root.game["tank" + _loc8_] != myTank && _root.game["tank" + _loc8_].bulletsFired != _root.settingsMaxBullets)
         {
            var _loc20_ = _root.distancesForMaze[_root.tankFields[_loc8_].x][_root.tankFields[_loc8_].y];
            _loc5_ = 0;
            while(_loc5_ < _loc4_.length)
            {
               _loc3_ = 0;
               while(_loc3_ < _loc4_[0].length)
               {
                  _loc4_[_loc5_][_loc3_] += _loc20_[_loc5_][_loc3_];
                  _loc3_ = _loc3_ + 1;
               }
               _loc5_ = _loc5_ + 1;
            }
         }
         _loc8_ = _loc8_ + 1;
      }
      if(_loc4_[_loc6_][_loc7_] < LONGESTPATHTORUN)
      {
         _loc46_ = {goal:"runAway",dist:_loc4_,period:10,priority:(LONGESTPATHTORUN - _loc4_[_loc6_][_loc7_]) / LONGESTPATHTORUN * COWARDNESS * (myTank.bulletsFired / _root.settingsMaxBullets),updateContinuously:false,id:goalId++};
         updateGoal(_loc46_);
      }
   }
   if(myTank.hitSomething)
   {
      stuckTime = Math.min(stuckTime + 1,MAXSTUCKTIME);
   }
   else
   {
      stuckTime = 0;
   }
   _loc46_ = {goal:"backAway",period:5,priority:stuckTime / (MAXSTUCKTIME - 0.1),updateContinuously:false,id:goalId++};
   updateGoal(_loc46_);
   if(_root.aliveCount > 1)
   {
      var _loc24_ = random(_root.TANKS);
      while(_root.game["tank" + _loc24_] == myTank || !_root.game["tank" + _loc24_].alive)
      {
         _loc24_ = random(_root.TANKS);
      }
      if(_root.game["tank" + _loc24_] != myTank)
      {
         _loc46_ = {goal:"driveTo",period:10,priority:IDLEDRIVETOWARDENEMYPRIORITY,x:_root.tankFields[_loc24_].x,y:_root.tankFields[_loc24_].y,updateContinuously:false,id:goalId++};
         updateGoal(_loc46_);
      }
   }
   if(oldGoal.id != myGoal.id)
   {
      switch(myGoal.goal)
      {
         case "shootAfter":
            trace("Goal: Shoot after " + myGoal.target);
            currentAggresiveness = Math.max(0,currentAggresiveness - 0.2);
            break;
         case "sprayBullets":
            trace("Goal: Spray bullets at " + myGoal.target);
            currentAggresiveness = Math.max(0,currentAggresiveness - 0.1);
            break;
         case "detonate":
            currentAggresiveness = Math.max(0,currentAggresiveness - 0.1);
            break;
         case "runAway":
            trace("Goal: Run away");
            break;
         case "backAway":
            break;
         case "driveTo":
            trace("Goal: Drive to " + myGoal.x + ", " + myGoal.y);
            break;
         case "dodgeBullet":
            trace("Goal: Dodge bullet at " + myGoal.x + ", " + myGoal.y);
            break;
         case "dodgeFragbomb":
         case "dodgeLaser":
         case "driveAfter":
         case "goForCrate":
      }
      return true;
   }
   currentAggresiveness = Math.min(AGGRESIVENESS,currentAggresiveness + AGGRESIVENESS / 50);
   return myGoal.updateContinuously;
}
function decideActionsToAchieveGoal()
{
   myActionsForGoal = new Array();
   var _loc9_ = Math.floor(myTank._x / _root.SCALE);
   var _loc10_ = Math.floor(myTank._y / _root.SCALE);
   switch(myGoal.goal)
   {
      case "shootAfter":
         var _loc6_ = myTank._rotation;
         var _loc11_ = false;
         var _loc8_ = _root.BULLETLIFETIME;
         var _loc5_ = _root.MOVIEWIDTH + _root.MOVIEHEIGHT;
         var _loc2_ = myTank._rotation;
         var _loc13_ = myGoal.target.x - myTank.x;
         var _loc12_ = myGoal.target.y - myTank.y;
         var _loc34_ = Math.sqrt(_loc13_ * _loc13_ + _loc12_ * _loc12_);
         var _loc27_ = checkPathForCollision(myTank.x,myTank.y,_loc13_ / _loc34_,_loc12_ / _loc34_,1,Math.ceil(_loc34_),Math.ceil(_loc34_));
         if(_loc27_ == undefined)
         {
            _loc11_ = true;
            _loc5_ = 0;
            if(_loc13_ != 0)
            {
               if(_loc13_ > 0)
               {
                  _loc6_ = 90 + Math.atan(_loc12_ / _loc13_) * 180 / 3.141592653589793;
               }
               else
               {
                  _loc6_ = -90 + Math.atan(_loc12_ / _loc13_) * 180 / 3.141592653589793;
               }
            }
            else if(_loc12_ > 0)
            {
               _loc6_ = 180;
            }
            else if(_loc12_ < 0)
            {
               _loc6_ = 0;
            }
            else
            {
               _loc6_ = _loc2_;
            }
            trace("Set shot to be a direct hitter with angle " + _loc6_);
         }
         if(!_loc11_)
         {
            var _loc4_ = 1;
            while(_loc4_ <= 3)
            {
               var _loc3_ = checkBulletPath(_loc2_);
               if(_loc3_.result == "HIT")
               {
                  _loc11_ = true;
                  if(_loc3_.time < _loc8_)
                  {
                     _loc8_ = _loc3_.time;
                     _loc5_ = 0;
                     _loc6_ = _loc2_;
                  }
               }
               else if(_loc3_.result == "NOTHING" && !_loc11_)
               {
                  if(_loc3_.closest < _loc5_)
                  {
                     _loc5_ = _loc3_.closest;
                     _loc6_ = _loc2_;
                  }
               }
               if(Math.random() < 0.5)
               {
                  _loc2_ += myTank.turnSpeed * _loc4_ * _loc4_;
               }
               else
               {
                  _loc2_ -= myTank.turnSpeed * _loc4_ * _loc4_;
               }
               if(_loc2_ < -180)
               {
                  _loc2_ = 360 + _loc2_;
               }
               if(_loc2_ > 180)
               {
                  _loc2_ -= 360;
               }
               _loc4_ = _loc4_ + 1;
            }
         }
         trace(myTank.currentWeapon);
         if(_loc11_ || _loc5_ < MAXCLOSESTDISTANCE / (myTank.currentWeapon != "laser" ? 1 : 2))
         {
            myActionsForGoal.push({action:"fireWeapon",delay:5});
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         else if(_loc6_ != myTank._rotation)
         {
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         else
         {
            _loc6_ = myTank._rotation + 180;
            if(_loc6_ > 180)
            {
               _loc6_ -= 360;
            }
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         break;
      case "sprayBullets":
         _loc6_ = myTank._rotation;
         _loc11_ = false;
         _loc8_ = _root.GATLINGLIFETIME;
         _loc5_ = _root.MOVIEWIDTH + _root.MOVIEHEIGHT;
         _loc2_ = myTank._rotation;
         _loc4_ = 1;
         while(_loc4_ <= 3)
         {
            _loc3_ = checkBulletPath(_loc2_);
            if(_loc3_.result == "HIT")
            {
               _loc11_ = true;
               if(_loc3_.time < _loc8_)
               {
                  _loc8_ = _loc3_.time;
                  _loc5_ = 0;
                  _loc6_ = _loc2_;
               }
            }
            else if(_loc3_.result == "NOTHING" && !foundGoodShot)
            {
               if(_loc3_.closest < _loc5_)
               {
                  _loc5_ = _loc3_.closest;
                  _loc6_ = _loc2_;
               }
            }
            if(Math.random() < 0.5)
            {
               _loc2_ += myTank.turnSpeed * _loc4_ * _loc4_;
            }
            else
            {
               _loc2_ -= myTank.turnSpeed * _loc4_ * _loc4_;
            }
            if(_loc2_ < -180)
            {
               _loc2_ = 360 + _loc2_;
            }
            if(_loc2_ > 180)
            {
               _loc2_ -= 360;
            }
            _loc4_ = _loc4_ + 1;
         }
         if(_loc11_ || _loc5_ < MAXCLOSESTDISTANCE)
         {
            myActionsForGoal.push({action:"fireWeapon",delay:75});
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         else if(_loc6_ != myTank._rotation)
         {
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         else
         {
            _loc6_ = myTank._rotation + 180;
            if(_loc6_ > 180)
            {
               _loc6_ -= 360;
            }
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
         }
         break;
      case "detonate":
         myActionsForGoal.push({action:"fireWeapon",delay:1});
         break;
      case "driveTo":
         var _loc26_ = _root.distancesForMaze[_loc9_][_loc10_];
         var _loc16_ = _root.getShortestPathWithDistances(_root.maze,_loc26_,_loc9_,_loc10_,myGoal.x,myGoal.y);
         pushActionsToFollowPath(_loc16_);
         break;
      case "runAway":
         _loc26_ = myGoal.dist;
         var _loc7_ = _root.followGradientPathWithDistancesAndDeadEnds(_root.maze,_loc26_,_root.deadEnds,_loc9_,_loc10_,5);
         pushActionsToFollowPath(_loc7_);
         break;
      case "backAway":
         myActionsForGoal.push({action:"driveToPos",x:(_loc9_ + 0.5) * _root.SCALE,y:(_loc10_ + 0.5) * _root.SCALE,canReverse:false});
         if(myTank.expandedHitCheck(myTank.hitPointsFront,1.1))
         {
            if(myTank.expandedHitCheck(myTank.hitPointsRear,1.1))
            {
               if(myTank.expandedHitCheck(myTank.hitPointsLeft,1.3000000000000005))
               {
                  myActionsForGoal.push({action:"backupAndTurn",dist:5,dir:"left"});
               }
               else
               {
                  myActionsForGoal.push({action:"backupAndTurn",dist:5,dir:"right"});
               }
            }
            else
            {
               myActionsForGoal.push({action:"backup",dist:3});
            }
         }
         else if(myTank.expandedHitCheck(myTank.hitPointsRear,1.1))
         {
            if(myTank.expandedHitCheck(myTank.hitPointsFront,1.1))
            {
               if(myTank.expandedHitCheck(myTank.hitPointsLeft,1.3000000000000005))
               {
                  myActionsForGoal.push({action:"backupAndTurn",dist:5,dir:"left"});
               }
               else
               {
                  myActionsForGoal.push({action:"backupAndTurn",dist:5,dir:"right"});
               }
            }
            else
            {
               myActionsForGoal.push({action:"forward",dist:3});
            }
         }
         else
         {
            myActionsForGoal.push({action:"backup",dist:3});
         }
         break;
      case "dodgeBullet":
         var _loc21_ = Math.floor(myGoal.x / _root.SCALE);
         var _loc23_ = Math.floor(myGoal.y / _root.SCALE);
         _loc7_ = _root.followGradientPathWithDistancesAndDeadEnds(_root.maze,_root.distancesForMaze[_loc21_][_loc23_],_root.deadEnds,_loc9_,_loc10_,5);
         if(myGoal.t < myGoal.maxTime / 3 && myGoal.dist < myGoal.maxDist / 5 || _loc7_.length <= 1)
         {
            if(_loc7_.length <= 1 && !(myGoal.t < myGoal.maxTime / 3 && myGoal.dist < myGoal.maxDist / 5))
            {
               trace("I was cornered!");
            }
            var _loc31_ = myTank._rotation;
            if(myGoal.dir.x != 0)
            {
               if(myGoal.dir.x > 0)
               {
                  _loc6_ = 90 + Math.atan(myGoal.dir.y / myGoal.dir.x) * 180 / 3.141592653589793;
               }
               else
               {
                  _loc6_ = -90 + Math.atan(myGoal.dir.y / myGoal.dir.x) * 180 / 3.141592653589793;
               }
            }
            else if(myGoal.dir.y > 0)
            {
               _loc6_ = 180;
            }
            else if(myGoal.dir.y < 0)
            {
               _loc6_ = 0;
            }
            else
            {
               _loc6_ = _loc31_;
            }
            if(Math.abs(_loc6_ - _loc31_) > 90 && Math.abs(_loc6_ - _loc31_) < 270)
            {
               _loc6_ += 180;
               if(_loc6_ > 180)
               {
                  _loc6_ -= 360;
               }
            }
            _loc6_ = Math.round(_loc6_ / myTank.turnSpeed) * myTank.turnSpeed;
            myActionsForGoal.push({action:"turnTo",angle:_loc6_});
            if(myGoal.dist < _root.SCALE / 4)
            {
               var _loc20_ = Math.sqrt(myGoal.dir.x * myGoal.dir.x + myGoal.dir.y * myGoal.dir.y);
               var _loc17_ = {x:(- myGoal.dir.y) / _loc20_,y:myGoal.dir.x / _loc20_};
               var _loc15_ = {x:myGoal.closest.x + _loc17_.x * _root.SCALE / 2,y:myGoal.closest.y + _loc17_.y * _root.SCALE / 2};
               var _loc14_ = {x:myGoal.closest.x - _loc17_.x * _root.SCALE / 2,y:myGoal.closest.y - _loc17_.y * _root.SCALE / 2};
               var _loc28_ = Math.sqrt((myTank.x - _loc15_.x) * (myTank.x - _loc15_.x) + (myTank.y - _loc15_.y) * (myTank.y - _loc15_.y));
               var _loc22_ = Math.sqrt((myTank.x - _loc14_.x) * (myTank.x - _loc14_.x) + (myTank.y - _loc14_.y) * (myTank.y - _loc14_.y));
               if(_loc28_ < _loc22_)
               {
                  myActionsForGoal.push({action:"driveToPos",x:_loc15_.x,y:_loc15_.y,canReverse:true});
               }
               else
               {
                  myActionsForGoal.push({action:"driveToPos",x:_loc14_.x,y:_loc14_.y,canReverse:true});
               }
            }
         }
         else
         {
            pushActionsToFollowPath(_loc7_);
         }
         tryToRetaliate();
         break;
      case "dodgeFragbomb":
         var _loc19_ = Math.floor(myGoal.frag.x / _root.SCALE);
         var _loc18_ = Math.floor(myGoal.frag.y / _root.SCALE);
         _loc7_ = _root.followGradientPathWithDistancesAndDeadEnds(_root.maze,_root.distancesForMaze[_loc19_][_loc18_],_root.deadEnds,_loc9_,_loc10_,5);
         if(_loc7_.length > 1)
         {
            pushActionsToFollowPath(_loc7_);
         }
         else
         {
            _loc7_ = _root.followGradientPathWithDistances(_root.maze,_root.distancesForMaze[_loc19_][_loc18_],_loc9_,_loc10_,5);
            pushActionsToFollowPath(_loc7_);
         }
         tryToRetaliate();
         break;
      case "dodgeLaser":
         var _loc25_ = Math.floor(myGoal.owner.x / _root.SCALE);
         var _loc24_ = Math.floor(myGoal.owner.y / _root.SCALE);
         _loc7_ = _root.followGradientPathWithDistancesAndDeadEnds(_root.maze,_root.distancesForMaze[_loc25_][_loc24_],_root.deadEnds,_loc9_,_loc10_,2);
         pushActionsToFollowPath(_loc7_);
         tryToRetaliate();
         break;
      case "goForCrate":
         _loc26_ = _root.distancesForMaze[_loc9_][_loc10_];
         _loc16_ = _root.getShortestPathWithDistances(_root.maze,_loc26_,_loc9_,_loc10_,myGoal.x,myGoal.y);
         myActionsForGoal.push({action:"driveToPos",x:(_loc16_[_loc16_.length - 1].x + 0.5) * _root.SCALE,y:(_loc16_[_loc16_.length - 1].y + 0.5) * _root.SCALE,canReverse:true});
         pushActionsToFollowPath(_loc16_);
         break;
      case "idle":
         myActionsForGoal.push({action:"idle"});
   }
}
function setInputToDoActions()
{
   var _loc6_ = Math.floor(myTank._x / _root.SCALE);
   var _loc7_ = Math.floor(myTank._y / _root.SCALE);
   action = myActionsForGoal.pop();
   switch(action.action)
   {
      case "driveToField":
         if(Math.abs(myTank._x - (action.x + 0.5) * _root.SCALE) > _root.SCALE / 3 || Math.abs(myTank._y - (action.y + 0.5) * _root.SCALE) > _root.SCALE / 3)
         {
            myActionsForGoal.push(action);
         }
         break;
      case "turnTo":
         if(Math.abs(myTank._rotation - action.angle) >= myTank.turnSpeed)
         {
            myActionsForGoal.push(action);
         }
         break;
      case "fireWeapon":
         if(action.delay != 0)
         {
            action.delay--;
            myActionsForGoal.push(action);
         }
         break;
      case "driveToPos":
         if(Math.abs(myTank._x - action.x) > _root.SCALE / 4 || Math.abs(myTank._y - action.y) > _root.SCALE / 4)
         {
            myActionsForGoal.push(action);
         }
         break;
      case "forward":
         if(action.dist != 0)
         {
            action.dist--;
            myActionsForGoal.push(action);
         }
         break;
      case "forwardAndTurn":
         if(action.dist != 0)
         {
            action.dist--;
            myActionsForGoal.push(action);
         }
      case "backup":
         if(action.dist != 0)
         {
            action.dist--;
            myActionsForGoal.push(action);
         }
         break;
      case "backupAndTurn":
         if(action.dist != 0)
         {
            action.dist--;
            myActionsForGoal.push(action);
         }
         break;
      case "idle":
         myActionsForGoal.push(action);
   }
   action = myActionsForGoal[myActionsForGoal.length - 1];
   switch(action.action)
   {
      case "driveToField":
         var _loc3_ = myTank._rotation;
         var _loc2_ = undefined;
         if(_loc6_ > action.x)
         {
            _loc2_ = -90;
         }
         else if(_loc6_ < action.x)
         {
            _loc2_ = 90;
         }
         else if(_loc7_ > action.y)
         {
            _loc2_ = 0;
         }
         else if(_loc7_ < action.y)
         {
            _loc2_ = 180;
         }
         else
         {
            _loc2_ = _loc3_;
         }
         if(_loc2_ > _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = true;
               myTank.turnRight = false;
            }
            else
            {
               myTank.turnLeft = false;
               myTank.turnRight = true;
            }
         }
         else if(_loc2_ < _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = false;
               myTank.turnRight = true;
            }
            else
            {
               myTank.turnLeft = true;
               myTank.turnRight = false;
            }
         }
         else
         {
            myTank.turnLeft = false;
            myTank.turnRight = false;
         }
         if(Math.abs(_loc2_ - _loc3_) > 90 && Math.abs(_loc2_ - _loc3_) < 270)
         {
            myTank.forward = false;
            myTank.backup = false;
         }
         else
         {
            myTank.forward = true;
            myTank.backup = false;
         }
         myTank.fire = false;
         break;
      case "turnTo":
         _loc3_ = myTank._rotation;
         _loc2_ = action.angle;
         if(_loc2_ > _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = true;
               myTank.turnRight = false;
            }
            else
            {
               myTank.turnLeft = false;
               myTank.turnRight = true;
            }
         }
         else if(_loc2_ < _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = false;
               myTank.turnRight = true;
            }
            else
            {
               myTank.turnLeft = true;
               myTank.turnRight = false;
            }
         }
         else
         {
            myTank.turnLeft = false;
            myTank.turnRight = false;
         }
         myTank.forward = false;
         myTank.backup = false;
         myTank.fire = false;
         break;
      case "fireWeapon":
         myTank.turnLeft = false;
         myTank.turnRight = false;
         myTank.forward = false;
         myTank.backup = false;
         myTank.fire = true;
         break;
      case "driveToPos":
         _loc3_ = myTank._rotation;
         var _loc5_ = false;
         var _loc4_ = {x:action.x - myTank._x,y:action.y - myTank._y};
         if(_loc4_.x != 0)
         {
            if(_loc4_.x > 0)
            {
               _loc2_ = 90 + Math.atan(_loc4_.y / _loc4_.x) * 180 / 3.141592653589793;
            }
            else
            {
               _loc2_ = -90 + Math.atan(_loc4_.y / _loc4_.x) * 180 / 3.141592653589793;
            }
         }
         else if(_loc4_.y > 0)
         {
            _loc2_ = 180;
         }
         else if(_loc4_.y < 0)
         {
            _loc2_ = 0;
         }
         else
         {
            _loc2_ = _loc3_;
         }
         _loc2_ = myTank.turnSpeed * Math.round(_loc2_ / myTank.turnSpeed);
         if(action.canReverse)
         {
            if(Math.abs(_loc2_ - _loc3_) > 90 && Math.abs(_loc2_ - _loc3_) < 270)
            {
               _loc5_ = true;
               _loc2_ += 180;
               if(_loc2_ > 180)
               {
                  _loc2_ -= 360;
               }
            }
         }
         if(_loc2_ > _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = Math.abs(_loc2_ - _loc3_) < 360 - myTank.turnSpeed ? true : false;
               myTank.turnRight = false;
            }
            else
            {
               myTank.turnLeft = false;
               myTank.turnRight = Math.abs(_loc2_ - _loc3_) > myTank.turnSpeed ? true : false;
            }
         }
         else if(_loc2_ < _loc3_)
         {
            if(Math.abs(_loc2_ - _loc3_) > 180)
            {
               myTank.turnLeft = false;
               myTank.turnRight = Math.abs(_loc2_ - _loc3_) < 360 - myTank.turnSpeed ? true : false;
            }
            else
            {
               myTank.turnLeft = Math.abs(_loc2_ - _loc3_) > myTank.turnSpeed ? true : false;
               myTank.turnRight = false;
            }
         }
         else
         {
            myTank.turnLeft = false;
            myTank.turnRight = false;
         }
         if(Math.abs(_loc2_ - _loc3_) > 45 && Math.abs(_loc2_ - _loc3_) < 315)
         {
            myTank.forward = false;
            myTank.backup = false;
         }
         else
         {
            myTank.forward = !_loc5_;
            myTank.backup = _loc5_;
         }
         myTank.fire = false;
         break;
      case "forward":
         myTank.turnLeft = false;
         myTank.turnRight = false;
         myTank.forward = true;
         myTank.backup = false;
         myTank.fire = false;
         break;
      case "forwardAndTurn":
         myTank.turnLeft = action.dir == "left";
         myTank.turnRight = action.dir == "right";
         myTank.forward = true;
         myTank.backup = false;
         myTank.fire = false;
         break;
      case "backup":
         myTank.turnLeft = false;
         myTank.turnRight = false;
         myTank.forward = false;
         myTank.backup = true;
         myTank.fire = false;
         break;
      case "backupAndTurn":
         myTank.turnLeft = action.dir == "left";
         myTank.turnRight = action.dir == "right";
         myTank.forward = false;
         myTank.backup = true;
         myTank.fire = false;
         break;
      case "idle":
         myTank.turnLeft = false;
         myTank.turnRight = false;
         myTank.forward = false;
         myTank.backup = false;
         myTank.fire = false;
         break;
      default:
         myTank.turnLeft = false;
         myTank.turnRight = false;
         myTank.forward = false;
         myTank.backup = false;
         myTank.fire = false;
         myGoal.period = 0;
   }
}
var myTank;
var myGoal = {goal:"idle",priority:0,period:15,id:0,updateContinuously:true};
var myActionsForGoal;
AGGRESIVENESS = 0.5;
COWARDNESS = 0.7000000000000001;
GREEDY = 1;
LONGESTPATHTOSHOOT = 7;
LONGESTPATHTONOTHESITATETOSHOOT = 2;
FRAGBOMBSAFETYDIST = 3 * _root.SCALE;
FRAGBOMBDETONATEDIST = 3 * _root.SCALE;
LONGESTPATHTORUN = 10;
MAXSTUCKTIME = 1;
stuckTime = 0;
currentAggresiveness = AGGRESIVENESS;
IDLEDRIVETOWARDENEMYPRIORITY = 0.1;
IDLEDRIVEPRIORITY = 0.1;
MAXCLOSESTCELLDISTANCE = 2;
MAXCLOSESTDISTANCE = _root.SCALE * MAXCLOSESTCELLDISTANCE;
MAXTIMETODODGEBULLET = 75;
MAXDISTTODODGEBULLET = 4 * _root.SCALE;
MAXCELLDISTTODODGEBULLET = MAXTIMETODODGEBULLET * _root.BULLETSPEED / 50;
MAXCELLDISTTODODGEFRAGBOMB = 5;
MAXTIMETODODGEFRAGBOMBFRAGMENT = 50;
MAXDISTTODODGEFRAGBOMBFRAGMENT = 3 * _root.SCALE;
MAXCELLDISTTODODGEFRAGBOMBFRAGMENT = MAXTIMETODODGEFRAGBOMBFRAGMENT * (_root.FRAGSPEED + 4) / 50;
MAXTIMETODODGEGATLINGBULLET = 75;
MAXDISTTODODGEGATLINGBULLET = 3 * _root.SCALE;
MAXCELLDISTTODODGEGATLINGBULLET = MAXTIMETODODGEGATLINGBULLET * _root.GATLINGSPEED / 50;
MAXCELLDISTTODODGELASER = 2;
MAXCELLDISTTOGOFORCRATE = 10;
var goalId = 1;
