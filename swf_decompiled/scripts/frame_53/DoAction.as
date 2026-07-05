function createMaze(xsize, ysize)
{
   tempmaze = new Array(xsize + 1);
   var _loc2_ = 0;
   while(_loc2_ < tempmaze.length)
   {
      tempmaze[_loc2_] = new Array(ysize + 1);
      _loc2_ = _loc2_ + 1;
   }
   _loc2_ = 0;
   while(_loc2_ < tempmaze.length)
   {
      var _loc1_ = 0;
      while(_loc1_ < tempmaze[_loc2_].length)
      {
         tempmaze[_loc2_][_loc1_] = random(4);
         _loc1_ = _loc1_ + 1;
      }
      _loc2_ = _loc2_ + 1;
   }
   maze = new Array(xsize);
   _loc2_ = 0;
   while(_loc2_ < maze.length)
   {
      maze[_loc2_] = new Array(ysize);
      _loc2_ = _loc2_ + 1;
   }
   _loc2_ = 0;
   while(_loc2_ < maze.length)
   {
      _loc1_ = 0;
      while(_loc1_ < maze[_loc2_].length)
      {
         var _loc4_ = tempmaze[_loc2_][_loc1_ + 1] == 2 || tempmaze[_loc2_ + 1][_loc1_ + 1] == 0;
         var _loc3_ = tempmaze[_loc2_][_loc1_] == 1 || tempmaze[_loc2_][_loc1_ + 1] == 3;
         maze[_loc2_][_loc1_] = new Array(1,!_loc4_ ? 0 : 1,!_loc3_ ? 0 : 1);
         _loc1_ = _loc1_ + 1;
      }
      _loc2_ = _loc2_ + 1;
   }
   return maze;
}
function calcReachable(maze, startx, starty)
{
   _root.reachableIndex = new Array(maze.length);
   var _loc6_ = 0;
   while(_loc6_ < maze.length)
   {
      _root.reachableIndex[_loc6_] = new Array(maze[_loc6_].length);
      _loc6_ = _loc6_ + 1;
   }
   var _loc4_ = new Array();
   var _loc7_ = new Array();
   var _loc5_ = new Array();
   _loc5_.push({x:startx,y:starty});
   while(_loc5_.length > 0)
   {
      var _loc2_ = _loc5_.pop();
      reachableIndex[_loc2_.x][_loc2_.y] = _loc7_.length;
      _loc7_.push(_loc2_);
      _loc4_[_loc2_.x + _loc2_.y * maze.length] = true;
      if(maze[_loc2_.x][_loc2_.y][2] == 0 && _loc2_.x > 0)
      {
         if(_loc4_[_loc2_.x - 1 + _loc2_.y * maze.length] == undefined)
         {
            _loc4_[_loc2_.x - 1 + _loc2_.y * maze.length] = true;
            _loc5_.push({x:_loc2_.x - 1,y:_loc2_.y});
         }
      }
      if(maze[_loc2_.x + 1][_loc2_.y][2] == 0 && _loc2_.x < maze.length - 1)
      {
         if(_loc4_[_loc2_.x + 1 + _loc2_.y * maze.length] == undefined)
         {
            _loc4_[_loc2_.x + 1 + _loc2_.y * maze.length] = true;
            _loc5_.push({x:_loc2_.x + 1,y:_loc2_.y});
         }
      }
      if(maze[_loc2_.x][_loc2_.y - 1][1] == 0 && _loc2_.y > 0)
      {
         if(_loc4_[_loc2_.x + (_loc2_.y - 1) * maze.length] == undefined)
         {
            _loc4_[_loc2_.x + (_loc2_.y - 1) * maze.length] = true;
            _loc5_.push({x:_loc2_.x,y:_loc2_.y - 1});
         }
      }
      if(maze[_loc2_.x][_loc2_.y][1] == 0 && _loc2_.y < maze[_loc2_.x].length - 1)
      {
         if(_loc4_[_loc2_.x + (_loc2_.y + 1) * maze.length] == undefined)
         {
            _loc4_[_loc2_.x + (_loc2_.y + 1) * maze.length] = true;
            _loc5_.push({x:_loc2_.x,y:_loc2_.y + 1});
         }
      }
   }
   return _loc7_;
}
function findDeadEnds(maze, reachable)
{
   var _loc2_ = new Array(maze.length);
   var _loc6_ = 0;
   while(_loc6_ < _loc2_.length)
   {
      _loc2_[_loc6_] = new Array(maze[_loc6_].length);
      _loc6_ = _loc6_ + 1;
   }
   var _loc8_ = new Array();
   _loc6_ = 0;
   while(_loc6_ < reachable.length)
   {
      _loc8_.push(reachable[_loc6_]);
      _loc2_[reachable[_loc6_].x][reachable[_loc6_].y] = 0;
      _loc6_ = _loc6_ + 1;
   }
   while(_loc8_.length > 0)
   {
      var _loc1_ = _loc8_.pop();
      if(!_loc2_[_loc1_.x][_loc1_.y])
      {
         var _loc7_ = undefined;
         var _loc5_ = 0;
         var _loc4_ = MAXDEADENDPENALTY;
         if(maze[_loc1_.x][_loc1_.y][2] == 0 && _loc1_.x > 0 && !_loc2_[_loc1_.x - 1][_loc1_.y])
         {
            _loc7_ = {x:_loc1_.x - 1,y:_loc1_.y};
            _loc5_ = _loc5_ + 1;
         }
         else if(maze[_loc1_.x][_loc1_.y][2] == 0 && _loc1_.x > 0)
         {
            _loc4_ = Math.max(1,Math.min(_loc2_[_loc1_.x - 1][_loc1_.y] - 1,_loc4_));
         }
         if(maze[_loc1_.x + 1][_loc1_.y][2] == 0 && _loc1_.x < maze.length - 1 && !_loc2_[_loc1_.x + 1][_loc1_.y])
         {
            _loc7_ = {x:_loc1_.x + 1,y:_loc1_.y};
            _loc5_ = _loc5_ + 1;
         }
         else if(maze[_loc1_.x + 1][_loc1_.y][2] == 0 && _loc1_.x < maze.length - 1)
         {
            _loc4_ = Math.max(1,Math.min(_loc2_[_loc1_.x + 1][_loc1_.y] - 1,_loc4_));
         }
         if(maze[_loc1_.x][_loc1_.y - 1][1] == 0 && _loc1_.y > 0 && !_loc2_[_loc1_.x][_loc1_.y - 1])
         {
            _loc7_ = {x:_loc1_.x,y:_loc1_.y - 1};
            _loc5_ = _loc5_ + 1;
         }
         else if(maze[_loc1_.x][_loc1_.y - 1][1] == 0 && _loc1_.y > 0)
         {
            _loc4_ = Math.max(1,Math.min(_loc2_[_loc1_.x][_loc1_.y - 1] - 1,_loc4_));
         }
         if(maze[_loc1_.x][_loc1_.y][1] == 0 && _loc1_.y < maze[_loc1_.x].length - 1 && !_loc2_[_loc1_.x][_loc1_.y + 1])
         {
            _loc7_ = {x:_loc1_.x,y:_loc1_.y + 1};
            _loc5_ = _loc5_ + 1;
         }
         else if(maze[_loc1_.x][_loc1_.y][1] == 0 && _loc1_.y < maze[_loc1_.x].length - 1)
         {
            _loc4_ = Math.max(1,Math.min(_loc2_[_loc1_.x][_loc1_.y + 1] - 1,_loc4_));
         }
         if(_loc5_ == 1)
         {
            _loc2_[_loc1_.x][_loc1_.y] = _loc4_;
            _loc8_.push(_loc7_);
         }
         if(_loc5_ == 0)
         {
            _loc2_[_loc1_.x][_loc1_.y] = _loc4_;
         }
      }
   }
   return _loc2_;
}
function calcDistances(maze, startx, starty)
{
   var _loc3_ = new Array(maze.length);
   var _loc6_ = 0;
   while(_loc6_ < _loc3_.length)
   {
      _loc3_[_loc6_] = new Array(maze[_loc6_].length);
      _loc6_ = _loc6_ + 1;
   }
   var _loc4_ = new Array();
   var _loc5_ = new Array();
   _loc5_.push({x:startx,y:starty});
   var _loc7_ = 0;
   _loc3_[startx][starty] = 0;
   while(_loc7_ < _loc5_.length)
   {
      var _loc1_ = _loc5_[_loc7_];
      _loc7_ = _loc7_ + 1;
      _loc4_[_loc1_.x + _loc1_.y * maze.length] = true;
      if(maze[_loc1_.x][_loc1_.y][2] == 0 && _loc1_.x > 0)
      {
         if(_loc4_[_loc1_.x - 1 + _loc1_.y * maze.length] == undefined)
         {
            _loc4_[_loc1_.x - 1 + _loc1_.y * maze.length] = true;
            _loc3_[_loc1_.x - 1][_loc1_.y] = _loc3_[_loc1_.x][_loc1_.y] + 1;
            _loc5_.push({x:_loc1_.x - 1,y:_loc1_.y});
         }
      }
      if(maze[_loc1_.x + 1][_loc1_.y][2] == 0 && _loc1_.x < maze.length - 1)
      {
         if(_loc4_[_loc1_.x + 1 + _loc1_.y * maze.length] == undefined)
         {
            _loc4_[_loc1_.x + 1 + _loc1_.y * maze.length] = true;
            _loc3_[_loc1_.x + 1][_loc1_.y] = _loc3_[_loc1_.x][_loc1_.y] + 1;
            _loc5_.push({x:_loc1_.x + 1,y:_loc1_.y});
         }
      }
      if(maze[_loc1_.x][_loc1_.y - 1][1] == 0 && _loc1_.y > 0)
      {
         if(_loc4_[_loc1_.x + (_loc1_.y - 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x + (_loc1_.y - 1) * maze.length] = true;
            _loc3_[_loc1_.x][_loc1_.y - 1] = _loc3_[_loc1_.x][_loc1_.y] + 1;
            _loc5_.push({x:_loc1_.x,y:_loc1_.y - 1});
         }
      }
      if(maze[_loc1_.x][_loc1_.y][1] == 0 && _loc1_.y < maze[_loc1_.x].length - 1)
      {
         if(_loc4_[_loc1_.x + (_loc1_.y + 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x + (_loc1_.y + 1) * maze.length] = true;
            _loc3_[_loc1_.x][_loc1_.y + 1] = _loc3_[_loc1_.x][_loc1_.y] + 1;
            _loc5_.push({x:_loc1_.x,y:_loc1_.y + 1});
         }
      }
      if(maze[_loc1_.x][_loc1_.y][1] == 0 && maze[_loc1_.x][_loc1_.y][2] == 0 && maze[_loc1_.x - 1][_loc1_.y][1] == 0 && maze[_loc1_.x][_loc1_.y + 1][2] == 0 && _loc1_.x > 0 && _loc1_.y < maze[_loc1_.x].length - 1)
      {
         if(_loc4_[_loc1_.x - 1 + (_loc1_.y + 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x - 1 + (_loc1_.y + 1) * maze.length] = true;
            _loc3_[_loc1_.x - 1][_loc1_.y + 1] = _loc3_[_loc1_.x][_loc1_.y] + 1.4142135623730951;
            _loc5_.push({x:_loc1_.x - 1,y:_loc1_.y + 1});
         }
      }
      if(maze[_loc1_.x][_loc1_.y][1] == 0 && maze[_loc1_.x + 1][_loc1_.y][2] == 0 && maze[_loc1_.x + 1][_loc1_.y][1] == 0 && maze[_loc1_.x + 1][_loc1_.y + 1][2] == 0 && _loc1_.x < maze.length - 1 && _loc1_.y < maze[_loc1_.x].length - 1)
      {
         if(_loc4_[_loc1_.x + 1 + (_loc1_.y + 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x + 1 + (_loc1_.y + 1) * maze.length] = true;
            _loc3_[_loc1_.x + 1][_loc1_.y + 1] = _loc3_[_loc1_.x][_loc1_.y] + 1.4142135623730951;
            _loc5_.push({x:_loc1_.x + 1,y:_loc1_.y + 1});
         }
      }
      if(maze[_loc1_.x][_loc1_.y][2] == 0 && maze[_loc1_.x][_loc1_.y - 1][1] == 0 && maze[_loc1_.x][_loc1_.y - 1][2] == 0 && maze[_loc1_.x - 1][_loc1_.y - 1][1] == 0 && _loc1_.x > 0 && _loc1_.y > 0)
      {
         if(_loc4_[_loc1_.x - 1 + (_loc1_.y - 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x - 1 + (_loc1_.y - 1) * maze.length] = true;
            _loc3_[_loc1_.x - 1][_loc1_.y - 1] = _loc3_[_loc1_.x][_loc1_.y] + 1.4142135623730951;
            _loc5_.push({x:_loc1_.x - 1,y:_loc1_.y - 1});
         }
      }
      if(maze[_loc1_.x + 1][_loc1_.y][2] == 0 && maze[_loc1_.x][_loc1_.y - 1][1] == 0 && maze[_loc1_.x + 1][_loc1_.y - 1][1] == 0 && maze[_loc1_.x + 1][_loc1_.y - 1][2] == 0 && _loc1_.x < maze.length - 1 && _loc1_.y > 0)
      {
         if(_loc4_[_loc1_.x + 1 + (_loc1_.y - 1) * maze.length] == undefined)
         {
            _loc4_[_loc1_.x + 1 + (_loc1_.y - 1) * maze.length] = true;
            _loc3_[_loc1_.x + 1][_loc1_.y - 1] = _loc3_[_loc1_.x][_loc1_.y] + 1.4142135623730951;
            _loc5_.push({x:_loc1_.x + 1,y:_loc1_.y - 1});
         }
      }
   }
   return _loc3_;
}
function getShortestPath(maze, startx, starty, endx, endy)
{
   var _loc1_ = calcDistances(maze,startx,starty);
   return getShortestPathWithDistances(maze,_loc1_,startx,starty,endx,endy);
}
function getShortestPathWithDistances(maze, distances, startx, starty, endx, endy)
{
   var _loc10_ = new Array();
   var _loc1_ = endx;
   var _loc2_ = endy;
   var _loc11_ = distances[_loc1_][_loc2_];
   var _loc4_ = _loc11_;
   var _loc7_ = endx;
   var _loc6_ = endy;
   do
   {
      _loc10_.push({x:_loc1_,y:_loc2_});
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_ - 1][_loc2_][1] == 0 && maze[_loc1_][_loc2_ + 1][2] == 0 && _loc1_ > 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ - 1][_loc2_ + 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ + 1];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_ + 1;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_ + 1][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_ + 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ + 1][_loc2_ + 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ + 1];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_ + 1;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_][_loc2_ - 1][2] == 0 && maze[_loc1_ - 1][_loc2_ - 1][1] == 0 && _loc1_ > 0 && _loc2_ > 0 && distances[_loc1_ - 1][_loc2_ - 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ - 1];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_ - 1;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ > 0 && distances[_loc1_ + 1][_loc2_ - 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ - 1];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_ - 1;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && _loc1_ > 0 && distances[_loc1_ - 1][_loc2_] < _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && _loc1_ < maze.length - 1 && distances[_loc1_ + 1][_loc2_] < _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_;
      }
      if(maze[_loc1_][_loc2_ - 1][1] == 0 && _loc2_ > 0 && distances[_loc1_][_loc2_ - 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ - 1];
         _loc7_ = _loc1_;
         _loc6_ = _loc2_ - 1;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_][_loc2_ + 1] < _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ + 1];
         _loc7_ = _loc1_;
         _loc6_ = _loc2_ + 1;
      }
      _loc11_ = _loc4_;
      _loc1_ = _loc7_;
      _loc2_ = _loc6_;
   }
   while(_loc1_ != startx || _loc2_ != starty);
   
   _loc10_.reverse();
   return _loc10_;
}
function optimizeShortestPath(maze, path)
{
   var _loc2_ = path.pop();
   var _loc4_ = path.pop();
   var _loc3_ = path[path.length - 1];
   if(checkClearPath(maze,_loc3_,_loc2_))
   {
      path.push(_loc2_);
   }
   else
   {
      path.push(_loc4_);
      path.push(_loc2_);
   }
}
function checkClearPath(maze, start, end)
{
   var _loc10_ = Math.sqrt((start.x - end.x) * (start.x - end.x) + (start.y - end.y) * (start.y - end.y));
   var _loc6_ = (end.x - start.x) / _loc10_;
   var _loc5_ = (end.y - start.y) / _loc10_;
   var _loc8_ = start.x + 0.5;
   var _loc7_ = start.y + 0.5;
   var _loc1_ = Math.floor(_loc8_);
   var _loc2_ = Math.floor(_loc7_);
   while(_loc1_ != end.x || _loc2_ != end.y)
   {
      if(_loc6_ < 0 && _loc5_ > 0 && _loc1_ > end.x && _loc2_ < end.y)
      {
         if(!(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_ - 1][_loc2_][1] == 0 && maze[_loc1_][_loc2_ + 1][2] == 0 && _loc1_ > 0 && _loc2_ < maze[_loc1_].length - 1))
         {
            return false;
         }
      }
      if(_loc6_ > 0 && _loc5_ > 0 && _loc1_ < end.x && _loc2_ < end.y)
      {
         if(!(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_ + 1][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_ + 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ < maze[_loc1_].length - 1))
         {
            return false;
         }
      }
      if(_loc6_ < 0 && _loc5_ < 0 && _loc1_ > end.x && _loc2_ > end.y)
      {
         if(!(maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_][_loc2_ - 1][2] == 0 && maze[_loc1_ - 1][_loc2_ - 1][1] == 0 && _loc1_ > 0 && _loc2_ > 0))
         {
            return false;
         }
      }
      if(_loc6_ > 0 && _loc5_ < 0 && _loc1_ < end.x && _loc2_ > end.y)
      {
         if(!(maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ > 0))
         {
            return false;
         }
      }
      if(_loc6_ < 0 && _loc5_ == 0)
      {
         if(!(maze[_loc1_][_loc2_][2] == 0 && _loc1_ > 0))
         {
            return false;
         }
      }
      if(_loc6_ > 0 && _loc5_ == 0)
      {
         if(!(maze[_loc1_ + 1][_loc2_][2] == 0 && _loc1_ < maze.length - 1))
         {
            return false;
         }
      }
      if(_loc6_ == 0 && _loc5_ < 0)
      {
         if(!(maze[_loc1_][_loc2_ - 1][1] == 0 && _loc2_ > 0))
         {
            return false;
         }
      }
      if(_loc6_ == 0 && _loc5_ > 0)
      {
         if(!(maze[_loc1_][_loc2_][1] == 0 && _loc2_ < maze[_loc1_].length - 1))
         {
            return false;
         }
      }
      _loc8_ += _loc6_;
      _loc7_ += _loc5_;
      _loc1_ = Math.floor(_loc8_);
      _loc2_ = Math.floor(_loc7_);
   }
   return true;
}
function followGradientPathWithDistances(maze, distances, startx, starty, maxLength)
{
   var _loc11_ = new Array();
   var _loc1_ = startx;
   var _loc2_ = starty;
   var _loc7_ = startx;
   var _loc6_ = starty;
   var _loc12_ = distances[_loc1_][_loc2_];
   var _loc4_ = _loc12_;
   do
   {
      var foundPlace = false;
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_ - 1][_loc2_][1] == 0 && maze[_loc1_][_loc2_ + 1][2] == 0 && _loc1_ > 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ - 1][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ + 1];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_ + 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_ + 1][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_ + 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ + 1][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ + 1];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_ + 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_][_loc2_ - 1][2] == 0 && maze[_loc1_ - 1][_loc2_ - 1][1] == 0 && _loc1_ > 0 && _loc2_ > 0 && distances[_loc1_ - 1][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ - 1];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ > 0 && distances[_loc1_ + 1][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ - 1];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && _loc1_ > 0 && distances[_loc1_ - 1][_loc2_] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_];
         _loc7_ = _loc1_ - 1;
         _loc6_ = _loc2_;
         foundPlace = true;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && _loc1_ < maze.length - 1 && distances[_loc1_ + 1][_loc2_] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_];
         _loc7_ = _loc1_ + 1;
         _loc6_ = _loc2_;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_ - 1][1] == 0 && _loc2_ > 0 && distances[_loc1_][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ - 1];
         _loc7_ = _loc1_;
         _loc6_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ + 1];
         _loc7_ = _loc1_;
         _loc6_ = _loc2_ + 1;
         foundPlace = true;
      }
      _loc12_ = _loc4_;
      _loc1_ = _loc7_;
      _loc2_ = _loc6_;
      _loc11_.push({x:_loc1_,y:_loc2_});
      maxLength = maxLength - 1;
   }
   while(foundPlace && maxLength > 0);
   
   return _loc11_;
}
function followGradientPathWithDistancesAndDeadEnds(maze, distances, deadEnds, startx, starty, maxLength)
{
   var _loc16_ = new Array();
   var _loc1_ = startx;
   var _loc2_ = starty;
   var _loc8_ = startx;
   var _loc7_ = starty;
   var _loc9_ = distances[_loc1_][_loc2_] - deadEnds[_loc1_][_loc2_];
   var _loc4_ = _loc9_;
   do
   {
      var foundPlace = false;
      var _loc10_ = _loc9_;
      var _loc14_ = 0;
      var _loc11_ = 0;
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_ - 1][_loc2_][1] == 0 && maze[_loc1_][_loc2_ + 1][2] == 0 && _loc1_ > 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ - 1][_loc2_ + 1] - deadEnds[_loc1_ - 1][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ + 1] - deadEnds[_loc1_ - 1][_loc2_ + 1];
         _loc8_ = _loc1_ - 1;
         _loc7_ = _loc2_ + 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_ + 1][_loc2_][1] == 0 && maze[_loc1_ + 1][_loc2_ + 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_ + 1][_loc2_ + 1] - deadEnds[_loc1_ + 1][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ + 1] - deadEnds[_loc1_ + 1][_loc2_ + 1];
         _loc8_ = _loc1_ + 1;
         _loc7_ = _loc2_ + 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_][_loc2_ - 1][2] == 0 && maze[_loc1_ - 1][_loc2_ - 1][1] == 0 && _loc1_ > 0 && _loc2_ > 0 && distances[_loc1_ - 1][_loc2_ - 1] - deadEnds[_loc1_ - 1][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_ - 1] - deadEnds[_loc1_ - 1][_loc2_ - 1];
         _loc8_ = _loc1_ - 1;
         _loc7_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && maze[_loc1_][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][1] == 0 && maze[_loc1_ + 1][_loc2_ - 1][2] == 0 && _loc1_ < maze.length - 1 && _loc2_ > 0 && distances[_loc1_ + 1][_loc2_ - 1] - deadEnds[_loc1_ + 1][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_ - 1] - deadEnds[_loc1_ + 1][_loc2_ - 1];
         _loc8_ = _loc1_ + 1;
         _loc7_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][2] == 0 && _loc1_ > 0 && distances[_loc1_ - 1][_loc2_] - deadEnds[_loc1_ - 1][_loc2_] > _loc4_)
      {
         _loc4_ = distances[_loc1_ - 1][_loc2_] - deadEnds[_loc1_ - 1][_loc2_];
         _loc8_ = _loc1_ - 1;
         _loc7_ = _loc2_;
         foundPlace = true;
      }
      if(maze[_loc1_ + 1][_loc2_][2] == 0 && _loc1_ < maze.length - 1 && distances[_loc1_ + 1][_loc2_] - deadEnds[_loc1_ + 1][_loc2_] > _loc4_)
      {
         _loc4_ = distances[_loc1_ + 1][_loc2_] - deadEnds[_loc1_ + 1][_loc2_];
         _loc8_ = _loc1_ + 1;
         _loc7_ = _loc2_;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_ - 1][1] == 0 && _loc2_ > 0 && distances[_loc1_][_loc2_ - 1] - deadEnds[_loc1_][_loc2_ - 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ - 1] - deadEnds[_loc1_][_loc2_ - 1];
         _loc8_ = _loc1_;
         _loc7_ = _loc2_ - 1;
         foundPlace = true;
      }
      if(maze[_loc1_][_loc2_][1] == 0 && _loc2_ < maze[_loc1_].length - 1 && distances[_loc1_][_loc2_ + 1] - deadEnds[_loc1_][_loc2_ + 1] > _loc4_)
      {
         _loc4_ = distances[_loc1_][_loc2_ + 1] - deadEnds[_loc1_][_loc2_ + 1];
         _loc8_ = _loc1_;
         _loc7_ = _loc2_ + 1;
         foundPlace = true;
      }
      _loc9_ = _loc4_;
      _loc1_ = _loc8_;
      _loc2_ = _loc7_;
      _loc16_.push({x:_loc1_,y:_loc2_});
      maxLength = maxLength - 1;
   }
   while(foundPlace && maxLength > 0);
   
   return _loc16_;
}
function drawMaze(maze, scale)
{
   _root.game.createEmptyMovieClip("mazebg",-1000);
   _root.game.createEmptyMovieClip("mazemc",_root.game.getNextHighestDepth());
   var mazeWidth = Math.floor(maze.length * scale);
   var mazeHeight = Math.floor(maze[0].length * scale);
   var lineThickness = Math.floor(scale / 16);
   var edgeThickness = 1;
   with(_root.game.mazebg)
   {
      lineStyle(undefined,0,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            if(maze[x][y][0] != 0)
            {
               moveTo(Math.floor(x * scale) - lineThickness - edgeThickness,Math.floor(y * scale) - lineThickness - edgeThickness);
               beginFill(0,100);
               lineTo(Math.floor((x + 1) * scale) + lineThickness + edgeThickness,Math.floor(y * scale) - lineThickness - edgeThickness);
               lineTo(Math.floor((x + 1) * scale) + lineThickness + edgeThickness,Math.floor((y + 1) * scale) + lineThickness + edgeThickness);
               lineTo(Math.floor(x * scale) - lineThickness - edgeThickness,Math.floor((y + 1) * scale) + lineThickness + edgeThickness);
               endFill();
            }
            y++;
         }
         x++;
      }
      lineStyle(undefined,0,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            if(maze[x][y][0] != 0)
            {
               moveTo(Math.floor(x * scale),Math.floor(y * scale));
               beginFill(15132390,100);
               lineTo(Math.floor((x + 1) * scale),Math.floor(y * scale));
               lineTo(Math.floor((x + 1) * scale),Math.floor((y + 1) * scale));
               lineTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
               endFill();
            }
            y++;
         }
         x++;
      }
   }
   with(_root.game.mazemc)
   {
      lineStyle(2 * lineThickness,5066061,100,true,"none","square","square");
      var x = 0;
      while(x < maze.length)
      {
         var y = 0;
         while(y < maze[x].length)
         {
            moveTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
            if(maze[x][y][1] != 0)
            {
               lineTo(Math.floor((x + 1) * scale),Math.floor((y + 1) * scale));
            }
            moveTo(Math.floor(x * scale),Math.floor(y * scale));
            if(maze[x][y][2] != 0)
            {
               lineTo(Math.floor(x * scale),Math.floor((y + 1) * scale));
            }
            y++;
         }
         x++;
      }
      var x = 0;
      while(x < maze.length)
      {
         if(maze[x][0][0] != 0)
         {
            moveTo(Math.floor(x * scale),0);
            lineTo(Math.floor((x + 1) * scale),0);
         }
         if(maze[x][maze[x].length - 1][0] != 0)
         {
            moveTo(Math.floor(x * scale),Math.floor(maze[x].length * scale));
            lineTo(Math.floor((x + 1) * scale),Math.floor(maze[x].length * scale));
         }
         x++;
      }
      var y = 0;
      while(y < maze[0].length)
      {
         if(maze[0][y][0] != 0)
         {
            moveTo(0,Math.floor((y + 1) * scale));
            lineTo(0,Math.floor(y * scale));
         }
         if(maze[maze.length - 1][y][0] != 0)
         {
            moveTo(Math.floor(maze.length * scale),Math.floor((y + 1) * scale));
            lineTo(Math.floor(maze.length * scale),Math.floor(y * scale));
         }
         y++;
      }
   }
}
function drawReachable(points)
{
   _root.clear();
   _root.lineStyle(3,43520);
   var _loc2_ = 0;
   while(_loc2_ < points.length)
   {
      _root.moveTo(points[_loc2_].x * 10 + 5,points[_loc2_].y * 10 + 5);
      _root.lineTo(points[_loc2_].x * 10 + 5,points[_loc2_].y * 10 + 6);
      _loc2_ = _loc2_ + 1;
   }
}
function drawTanks(tanks)
{
   var _loc2_ = 0;
   while(_loc2_ < tanks.length)
   {
      _root.lineStyle(5,5570560 * _loc2_);
      _root.moveTo(tanks[_loc2_].x * 10 + 5,tanks[_loc2_].y * 10 + 5);
      _root.lineTo(tanks[_loc2_].x * 10 + 5,tanks[_loc2_].y * 10 + 6);
      _loc2_ = _loc2_ + 1;
   }
}
function drawDeadEnds(deadEnds)
{
   var _loc3_ = 0;
   while(_loc3_ < deadEnds.length)
   {
      var _loc2_ = 0;
      while(_loc2_ < deadEnds[_loc3_].length)
      {
         if(deadEnds[_loc3_][_loc2_])
         {
            var _loc4_ = 0;
            while(_loc4_ < deadEnds[_loc3_][_loc2_])
            {
               _root.game.mazebg.lineStyle(3,16711680);
               _root.game.mazebg.moveTo((_loc3_ + 0.5) * _root.SCALE - 4 * deadEnds[_loc3_][_loc2_] / 2 + 4 * _loc4_,(_loc2_ + 0.5) * _root.SCALE - 2 * deadEnds[_loc3_][_loc2_] / 2 + 2 * _loc4_);
               _root.game.mazebg.lineTo((_loc3_ + 0.5) * _root.SCALE + 1 - 4 * deadEnds[_loc3_][_loc2_] / 2 + 4 * _loc4_,(_loc2_ + 0.5) * _root.SCALE - 2 * deadEnds[_loc3_][_loc2_] / 2 + 2 * _loc4_);
               _loc4_ = _loc4_ + 1;
            }
         }
         else if(deadEnds[_loc3_][_loc2_] != undefined)
         {
            _root.game.mazebg.lineStyle(5,65280);
            _root.game.mazebg.moveTo((_loc3_ + 0.5) * _root.SCALE,(_loc2_ + 0.5) * _root.SCALE);
            _root.game.mazebg.lineTo((_loc3_ + 0.5) * _root.SCALE + 1,(_loc2_ + 0.5) * _root.SCALE);
         }
         _loc2_ = _loc2_ + 1;
      }
      _loc3_ = _loc3_ + 1;
   }
}
function drawPath(path)
{
   var _loc5_ = "path" + Math.random();
   var _loc6_ = 43520 + Math.random() * 17408;
   _root.game.mazebg.createEmptyMovieClip(_loc5_,_root.game.mazebg.getNextHighestDepth());
   _root.game.mazebg[_loc5_].life = 25;
   _root.game.mazebg[_loc5_].lineStyle(5,_loc6_);
   _root.game.mazebg[_loc5_].moveTo((path[0].x + 0.5) * _root.SCALE,(path[0].y + 0.5) * _root.SCALE);
   var _loc3_ = 1;
   while(_loc3_ < path.length)
   {
      _root.game.mazebg[_loc5_].lineTo((path[_loc3_].x + 0.5) * _root.SCALE,(path[_loc3_].y + 0.5) * _root.SCALE);
      _root.game.mazebg[_loc5_].lineStyle(15,_loc6_);
      _root.game.mazebg[_loc5_].lineTo((path[_loc3_].x + 0.5) * _root.SCALE + 1,(path[_loc3_].y + 0.5) * _root.SCALE);
      _root.game.mazebg[_loc5_].lineStyle(5,_loc6_);
      _loc3_ = _loc3_ + 1;
   }
   _root.game.mazebg[_loc5_].onEnterFrame = function()
   {
      this.life = this.life - 1;
      if(this.life < 0)
      {
         this.removeMovieClip();
      }
   };
}
function drawDir(x, y, xdir, ydir)
{
   var _loc3_ = "dir" + Math.random();
   _root.game.mazebg.createEmptyMovieClip(_loc3_,_root.game.mazebg.getNextHighestDepth());
   _root.game.mazebg[_loc3_].life = 25;
   _root.game.mazebg[_loc3_].lineStyle(5,16711680);
   _root.game.mazebg[_loc3_].moveTo(x,y);
   _root.game.mazebg[_loc3_].lineTo(x + 1,y);
   _root.game.mazebg[_loc3_].lineStyle(3,16711680);
   _root.game.mazebg[_loc3_].lineTo(x + xdir * 3,y + ydir * 3);
   _root.game.mazebg[_loc3_].onEnterFrame = function()
   {
      this.life = this.life - 1;
      if(this.life < 0)
      {
         this.removeMovieClip();
      }
   };
}
function deployTank(position, number, scale)
{
   _root.game.attachMovie("tank","tank" + number,_root.game.getNextHighestDepth());
   _root.game["tank" + number]._x = (position.x + 0.5) * scale + _root.game.mazemc._x;
   _root.game["tank" + number]._y = (position.y + 0.5) * scale + _root.game.mazemc._y;
   _root.game["tank" + number]._rotation = Math.floor(Math.random() * 32) * 11.25;
   _root.game["tank" + number]._xscale = 0.55 * scale;
   _root.game["tank" + number]._yscale = 0.55 * scale;
   _root.game["tank" + number].base.gotoAndStop(1);
   _root.game["tank" + number].turret.gotoAndStop(1);
   var _loc6_ = number;
   var _loc10_ = parseInt(_root.loginInfo["p" + (number + 1) + "bc"]);
   var _loc9_ = parseInt(_root.loginInfo["p" + (number + 1) + "tc"]);
   _root.game["tank" + number].baseColor = convertFromHexToRGB(_loc10_);
   _root.game["tank" + number].turretColor = convertFromHexToRGB(_loc9_);
   _root.game["tank" + number].username = _root.loginInfo["p" + (number + 1) + "n"];
   _loc6_ = _root.loginInfo.playerNumToControlNum[number];
   if(_root.loginInfo.actualRankedPlayers[number])
   {
      var _loc4_ = _root.game.createEmptyMovieClip("nameTag" + number,_root.game.getNextHighestDepth());
      _loc4_.trackTank = _root.game["tank" + number];
      _loc4_.lifetime = 50;
      _loc4_.name = _loc4_.createTextField("name",_loc4_.getNextHighestDepth(),0,-20,1,1);
      _loc4_.name.autoSize = "left";
      _loc4_.name.text = _root.loginInfo["p" + (number + 1) + "n"];
      var _loc5_ = new TextFormat();
      _loc5_.bold = true;
      _loc5_.font = "eurostile";
      _loc5_.size = 18;
      _loc5_.color = parseInt(_root.loginInfo["p" + (number + 1) + "bc"]);
      _loc4_.name.embedFonts = true;
      _loc4_.name.setTextFormat(_loc5_);
      _loc4_._x = _loc4_.trackTank._x - _loc4_.name._width / 2;
      _loc4_._y = Math.max(10,_loc4_.trackTank._y - 0.55 * scale);
      var _loc7_ = _loc4_.name.filters;
      _loc7_.push(new flash.filters.DropShadowFilter(3,90,0,0.5,5,5));
      _loc4_.name.filters = _loc7_;
      _loc4_.onEnterFrame = function()
      {
         this._x = this.trackTank._x - this.name._width / 2;
         this._y = Math.max(10,this.trackTank._y - 0.55 * scale);
         this.lifetime = this.lifetime - 1;
         if(this.lifetime <= 0)
         {
            this._alpha -= 10;
            if(this._alpha <= 0)
            {
               this.removeMovieClip();
            }
         }
         if(!this.trackTank.alive)
         {
            this.lifetime = 0;
         }
      };
   }
   switch(_loc6_)
   {
      case 0:
         _root.game["tank" + number].KEYTURNLEFT = 83;
         _root.game["tank" + number].KEYFORWARD = 69;
         _root.game["tank" + number].KEYTURNRIGHT = 70;
         _root.game["tank" + number].KEYBACKUP = 68;
         _root.game["tank" + number].KEYFIRE = 81;
         _root.game["tank" + number].mouseTank = false;
         break;
      case 1:
         _root.game["tank" + number].KEYTURNLEFT = 37;
         _root.game["tank" + number].KEYFORWARD = 38;
         _root.game["tank" + number].KEYTURNRIGHT = 39;
         _root.game["tank" + number].KEYBACKUP = 40;
         _root.game["tank" + number].KEYFIRE = 77;
         _root.game["tank" + number].mouseTank = false;
         break;
      case 2:
         _root.game["tank" + number].mouseTank = true;
         Mouse.hide();
         _root.attachMovie("scopeCross","scopeCross",_root.getNextHighestDepth());
         _root.attachMovie("scopeCircle","scopeCircle",_root.getNextHighestDepth());
         deltaX = _root.game.mazemc._xmouse - _root.game["tank" + number]._x;
         deltaY = _root.game.mazemc._ymouse - _root.game["tank" + number]._y;
         deltaLength = Math.sqrt(Math.pow(deltaX,2) + Math.pow(deltaY,2));
         _root.scopeCross._x = _root._xmouse;
         _root.scopeCross._y = _root._ymouse;
         if(deltaLength > 60)
         {
            _root.scopeCircle._x = _root.game._x + _root.game["tank" + number]._x + deltaX / deltaLength * 60;
            _root.scopeCircle._y = _root.game._y + _root.game["tank" + number]._y + deltaY / deltaLength * 60;
         }
         else
         {
            _root.scopeCircle._x = _root._xmouse;
            _root.scopeCircle._y = _root._ymouse;
         }
   }
   _root.game["tank" + number].scoreboard = _root["player" + (number + 1) + "ScoreBoard"];
   var _loc8_ = new Color(_root.game["tank" + number].base.background);
   _loc8_.setTint(_root.game["tank" + number].baseColor.r,_root.game["tank" + number].baseColor.g,_root.game["tank" + number].baseColor.b,_root.game["tank" + number].baseColor.a);
   _root.setEquipment(_root.game["tank" + number],"none");
   _root.setWeapon(_root.game["tank" + number],STARTWEAPON);
   if(_root.AIEnabled && number == 1)
   {
      AIDepth = _root.game["tank" + number].getNextHighestDepth();
      _root.game["tank" + number].attachMovie("tankTroubleAI","AI",AIDepth);
      trace(_root.game["tank" + number].AI);
      _root.game["tank" + number].AI.myTank = _root.game["tank" + number];
      _root.game["tank" + number].AI.myMaze = maze;
   }
}
function destroyTank(number)
{
   if(_root.soundOn)
   {
      _root.soundExplosion.start();
      _root.soundExplosion2.start();
   }
   _root.game["tank" + number].alive = false;
   _root.game["tank" + number]._visible = false;
   _root.aliveCount = _root.aliveCount - 1;
   _root.endCount = _root.NUMBEROFFRAMESBEFOREEND;
   _root.shake = Math.max(_root.MAXSHAKE,_root.shake + 7);
   var _loc5_ = 0;
   while(_loc5_ < _root.NUMBEROFSMOKECLOUDS)
   {
      _root.game.createEmptyMovieClip("smoke" + number + "-" + _loc5_,_root.game.getNextHighestDepth());
      s = _root.game["smoke" + number + "-" + _loc5_];
      s.lineStyle(15 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,40 + random(20));
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.yspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.x = _root.game["tank" + number]._x + s.xspeed * (random(6) + 1) + (random(2) - 1) * (_root.SCALE / 50);
      s.y = _root.game["tank" + number]._y + s.yspeed * (random(6) + 1) + (random(2) - 1) * (_root.SCALE / 50);
      s._x = s.x;
      s._y = s.y;
      s.onEnterFrame = function()
      {
         if(_root.frozen)
         {
            return undefined;
         }
         this._xscale += 2;
         this._yscale += 2;
         this._alpha -= 3 - Math.random() * 2;
         this.xspeed *= 0.93;
         this.yspeed *= 0.93;
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc5_ = _loc5_ + 1;
   }
   _loc5_ = 0;
   while(_loc5_ < _root.NUMBEROFFRAGMENTS)
   {
      _root.game.mazebg.createEmptyMovieClip("fragment" + number + "-" + _loc5_,_root.game.mazebg.getNextHighestDepth());
      f = _root.game.mazebg["fragment" + number + "-" + _loc5_];
      dir = Math.random() * 3.141592653589793 * 2;
      speed = Math.random() * 3 + 1;
      f.xspeed = Math.cos(dir) * (speed / 1.5) * (_root.SCALE / 50);
      f.yspeed = Math.sin(dir) * (speed / 1.5) * (_root.SCALE / 50);
      f.rotspeed = Math.random() * 120 - 60;
      f.active = true;
      f.smokenamebase = "smoke-fragment" + number + "-" + _loc5_;
      f.smokecounter = 0;
      f.hitPoints = new Array();
      f.lineStyle(1,0,100,false,"none");
      if(Math.random() > 0.4)
      {
         f.beginFill(parseInt(_root.loginInfo["p" + (number + 1) + "bc"]),100);
      }
      else
      {
         f.beginFill(parseInt(_root.loginInfo["p" + (number + 1) + "tc"]),100);
      }
      point1 = {x:random(10) - 5,y:random(10) - 5};
      point2 = {x:random(10) - 5,y:random(10) - 5};
      point3 = {x:random(10) - 5,y:random(10) - 5};
      point4 = {x:random(10) - 5,y:random(10) - 5};
      center = {x:(point1.x + point2.x + point3.x + point4.x) / 4,y:(point1.y + point2.y + point3.y + point4.y) / 4};
      f.moveTo((point1.x - center.x) * (_root.SCALE / 50),(point1.y - center.y) * (_root.SCALE / 50));
      f.lineTo((point2.x - center.x) * (_root.SCALE / 50),(point2.y - center.y) * (_root.SCALE / 50));
      f.hitPoints.push({x:(point2.x - center.x) * (_root.SCALE / 50),y:(point2.y - center.y) * (_root.SCALE / 50)});
      f.lineTo((point3.x - center.x) * (_root.SCALE / 50),(point3.y - center.y) * (_root.SCALE / 50));
      f.hitPoints.push({x:(point3.x - center.x) * (_root.SCALE / 50),y:(point3.y - center.y) * (_root.SCALE / 50)});
      f.lineTo((point4.x - center.x) * (_root.SCALE / 50),(point4.y - center.y) * (_root.SCALE / 50));
      f.hitPoints.push({x:(point4.x - center.x) * (_root.SCALE / 50),y:(point4.y - center.y) * (_root.SCALE / 50)});
      f.lineTo((point1.x - center.x) * (_root.SCALE / 50),(point1.y - center.y) * (_root.SCALE / 50));
      f.hitPoints.push({x:(point1.x - center.x) * (_root.SCALE / 50),y:(point1.y - center.y) * (_root.SCALE / 50)});
      f.endFill();
      f.spawnCounter = 0;
      f.x = _root.game["tank" + number]._x + f.xspeed * (random(5) + 2);
      f.y = _root.game["tank" + number]._y + f.yspeed * (random(5) + 2);
      f._x = f.x;
      f._y = f.y;
      f._rotation = random(360);
      f.onEnterFrame = function()
      {
         if(_root.frozen)
         {
            return undefined;
         }
         if(this.active)
         {
            this.spawnCounter = this.spawnCounter + 1;
            if(this.spawnCounter % 3)
            {
               _root.game.createEmptyMovieClip(this.smokenamebase + "-" + this.smokecounter,_root.game.getNextHighestDepth());
               s = _root.game[this.smokenamebase + "-" + this.smokecounter];
               this.smokecounter = this.smokecounter + 1;
               s.lineStyle(3 * (_root.SCALE / 50),Math.round(random(4)) * 1118481,30);
               s.moveTo(0,0);
               s.lineTo(0,1);
               s.xspeed = (Math.random() - 0.5) * (_root.SCALE / 50);
               s.yspeed = (Math.random() - 0.5) * (_root.SCALE / 50);
               s.x = this._x;
               s.y = this._y;
               s._x = s.x;
               s._y = s.y;
               s.onEnterFrame = function()
               {
                  if(_root.frozen)
                  {
                     return undefined;
                  }
                  this._xscale += 2;
                  this._yscale += 2;
                  this._alpha -= 3 - Math.random() * 3;
                  this.xspeed *= 0.9000000000000002;
                  this.yspeed *= 0.9000000000000002;
                  this.x += this.xspeed;
                  this.y += this.yspeed;
                  this._x = this.x;
                  this._y = this.y;
                  if(this._alpha <= 0)
                  {
                     this.removeMovieClip();
                  }
               };
            }
            this.x += this.xspeed;
            this.y += this.yspeed;
            this._x = this.x;
            this._y = this.y;
            this.xspeed *= 0.97;
            this.yspeed *= 0.97;
            this.rotspeed *= 0.97;
            this._rotation += this.rotspeed;
            if(this.hitCheck(this.hitPoints))
            {
               this.active = false;
            }
         }
         if(!this.active || Math.abs(this.xspeed) < 0.5 && Math.abs(this.yspeed) < 0.5)
         {
            this._alpha -= 5;
         }
         if(this._alpha <= 0)
         {
            this.active = false;
            this.removeMovieClip();
         }
      };
      f.hitCheck = function(points)
      {
         var _loc3_ = 0;
         while(_loc3_ < points.length)
         {
            point = {x:points[_loc3_].x,y:points[_loc3_].y};
            this.localToGlobal(point);
            if(_root.game.mazemc.hitTest(point.x,point.y,true))
            {
               return true;
            }
            _loc3_ = _loc3_ + 1;
         }
         return false;
      };
      _loc5_ = _loc5_ + 1;
   }
}
function lockedControl(owner, weapon)
{
   switch(weapon)
   {
      case "laser":
         return !owner.laserReady;
      case "deathRay":
         return !owner.deathRayReady;
      case "remote":
         return owner.remoteControlling;
      default:
         return false;
   }
}
function weaponReady(owner, weapon)
{
   switch(weapon)
   {
      case "bullet":
         return owner.bulletsFired < _root.settingsMaxBullets;
      case "laser":
         return owner.laserReady;
      case "frag":
         return true;
      case "gatling":
         return owner.gatlingReady;
      case "homing":
         return owner.homingReady;
      case "mine":
         return true;
      case "remote":
         return !owner.remoteControlling;
      case "electric":
         return owner.electricReady;
      case "deathRay":
         return owner.deathRayReady;
      default:
   }
}
function setWeapon(owner, weapon)
{
   owner.currentWeapon = weapon;
   switch(weapon)
   {
      case "bullet":
         owner.turret.gotoAndStop(1);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 16 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 16 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(1);
         break;
      case "laser":
         owner.turret.gotoAndStop(5);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 2.7,y:(- owner.base._height) / 2 * 1.2};
         owner.hitPointsFront[2] = {x:owner.base._width / 2.7,y:(- owner.base._height) / 2 * 1.2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2 * 1.2};
         owner.hitPointsFront[5] = {x:owner.base._width / 4,y:(- owner.base._height) / 2 * 1.2};
         owner.hitPointsFront[6] = {x:0,y:(- owner.turret._height) / 16 * 11.5};
         owner.hitPointsFront[7] = {x:0,y:(- owner.turret._height) / 16 * 9};
         owner.scoreboard.tankIcon.gotoAndStop(2);
         _root.setEquipment(owner,"aimer");
         break;
      case "frag":
         owner.turret.gotoAndStop(7);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 16 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 16 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(3);
         break;
      case "gatling":
         owner.turret.gotoAndStop(13);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 17 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 17 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(4);
         break;
      case "homing":
         owner.turret.gotoAndStop(17);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 16 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 16 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(6);
         break;
      case "mine":
         owner.turret.gotoAndStop(31);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 16 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 16 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(17);
         owner.minesLayed = 0;
         break;
      case "deathRay":
         owner.turret.gotoAndStop(20);
         owner.hitPointsFront = new Array();
         owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
         owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
         owner.hitPointsFront[4] = {x:(- owner.turret._width) / 6,y:(- owner.turret._height) / 17 * 11};
         owner.hitPointsFront[5] = {x:owner.turret._width / 6,y:(- owner.turret._height) / 17 * 11};
         owner.scoreboard.tankIcon.gotoAndStop(7);
         break;
      case "remote":
      case "electric":
   }
   var _loc3_ = new Color(owner.turret.background);
   _loc3_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   _loc3_ = new Color(owner.scoreboard.tankIcon.turretBackground);
   _loc3_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   _loc3_ = new Color(owner.scoreboard.tankIcon.turretBackground2);
   _loc3_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
}
function setEquipment(owner, equ)
{
   owner.equipment.removeMovieClip();
   owner.currentEquipment = equ;
   switch(equ)
   {
      case "none":
         break;
      case "aimer":
         owner.equipment = _root.addAimer(owner);
         break;
      case "shield":
         owner.equipment = _root.addShield(owner);
   }
}
function fireWeapon(owner, weapon)
{
   switch(weapon)
   {
      case "bullet":
         fireBullet(owner);
         break;
      case "laser":
         fireLaser(owner);
         break;
      case "frag":
         fireFrag(owner);
         break;
      case "gatling":
         fireGatling(owner);
         break;
      case "deathRay":
         fireDeathRay(owner);
         break;
      case "homing":
         fireHoming(owner);
         break;
      case "mine":
         layMine(owner);
         break;
      case "remote":
         fireRemote(owner);
         break;
      case "electric":
         fireElectric(owner);
   }
}
function fireBullet(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundBullet.start();
   }
   bulletDepth = _root.game.getNextHighestDepth();
   bulletName = "bullet" + bulletDepth;
   bullet = _root.game.attachMovie("bullet",bulletName,bulletDepth);
   owner.swapDepths(bullet);
   bullet.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet._x = bullet.x;
   bullet._y = bullet.y;
   bullet._xscale = 100 * (_root.SCALE / 50);
   bullet._yscale = 100 * (_root.SCALE / 50);
   bullet.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.lifetime = BULLETLIFETIME;
   bullet.deadly = BULLETDEADLY;
   bullet.owner = owner;
   owner.bulletsFired = owner.bulletsFired + 1;
}
function fireLaser(owner)
{
   if(_root.soundOn)
   {
      _root.soundLaser.start();
   }
   laserDepth = _root.game.getNextHighestDepth();
   laserName = "laser" + laserDepth;
   laser = _root.game.attachMovie("laser",laserName,laserDepth);
   owner.swapDepths(laser);
   laser._x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   laser._y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   laser.x = 0;
   laser.y = 0;
   laser.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * LASERSPEED / LASERHITCHECKINTERVALS * (_root.SCALE / 50);
   laser.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * LASERSPEED / LASERHITCHECKINTERVALS * (_root.SCALE / 50);
   laser.lifetime = LASERLIFETIME;
   laser.deadly = LASERDEADLY;
   laser.active = true;
   laser.owner = owner;
   laser.laserColor = 16711680 * owner.turretColor.r / 255 + 65280 * owner.turretColor.g / 255 + 255 * owner.turretColor.b / 255;
   laser.moveTo(0,0);
   owner.laserReady = false;
   setEquipment(owner,"none");
}
function fireFrag(owner)
{
   if(owner.fragFired && owner.alive)
   {
      owner.lastFrag.detonate();
   }
   else
   {
      owner.turret.play();
      if(_root.soundOn)
      {
         _root.soundFragment.start();
      }
      fragDepth = _root.game.getNextHighestDepth();
      fragName = "frag" + fragDepth;
      frag = _root.game.attachMovie("fragbomb",fragName,fragDepth);
      owner.swapDepths(frag);
      frag.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
      frag.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
      frag._x = frag.x;
      frag._y = frag.y;
      frag._xscale = 100 * (_root.SCALE / 50);
      frag._yscale = 100 * (_root.SCALE / 50);
      frag.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * FRAGSPEED / FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
      frag.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * FRAGSPEED / FRAGHITCHECKINTERVALS * (_root.SCALE / 50);
      frag.lifetime = FRAGLIFETIME;
      frag.deadly = FRAGDEADLY;
      frag.level = _root.FRAGLEVELS;
      frag.owner = owner;
      owner.lastFrag = frag;
      owner.fragFired = true;
   }
}
function fireGatling(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundGatlingMotorStart.start();
   }
   gatlingDepth = _root.game.getNextHighestDepth();
   gatlingName = "gatling" + gatlingDepth;
   gatling = _root.game.attachMovie("gatling",gatlingName,gatlingDepth);
   gatling.active = true;
   gatling.spinSpeed = 0;
   gatling.fireCounter = 0;
   gatling.bulletsLeft = GATLINGBULLETS;
   gatling.owner = owner;
   owner.gatlingReady = false;
}
function fireDeathRay(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundDeathRayCharge.start(1.3000000000000005);
   }
   deathRayDepth = _root.game.mazebg.getNextHighestDepth();
   deathRayName = "deathRay" + deathRayDepth;
   deathRay = _root.game.mazebg.attachMovie("deathRay",deathRayName,deathRayDepth);
   deathRay._x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   deathRay._y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   deathRay.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * (_root.SCALE / 16);
   deathRay.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * (_root.SCALE / 16);
   deathRay.active = false;
   deathRay.warmup = DEATHRAYWARMUPTIME;
   deathRay.lifetime = DEATHRAYLIFETIME;
   deathRay.owner = owner;
   owner.deathRayReady = false;
}
function fireHoming(owner)
{
   owner.turret.gotoAndStop(1);
   var _loc7_ = new Color(owner.turret.background);
   _loc7_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   if(_root.soundOn)
   {
      _root.soundHoming3.start();
   }
   homingDepth = _root.game.getNextHighestDepth();
   homingName = "homing" + homingDepth;
   homing = _root.game.attachMovie("homingbullet",homingName,homingDepth);
   owner.swapDepths(homing);
   homing.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   homing.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   homing._x = homing.x;
   homing._y = homing.y;
   homing._xscale = 100 * (_root.SCALE / 50);
   homing._yscale = 100 * (_root.SCALE / 50);
   homing.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * HOMINGSPEED / HOMINGHITCHECKINTERVALS * (_root.SCALE / 50);
   homing.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * HOMINGSPEED / HOMINGHITCHECKINTERVALS * (_root.SCALE / 50);
   var _loc8_ = new Color(homing.background);
   _loc8_.setRGB(16711680 * owner.turretColor.r / 255 + 65280 * owner.turretColor.g / 255 + 255 * owner.turretColor.b / 255);
   homing.lifetime = HOMINGLIFETIME;
   homing.deadly = HOMINGDEADLY;
   homing.homing = false;
   homing.startuptime = HOMINGSTARTUPTIME;
   homing.soundCounter = 0;
   homing.owner = owner;
   homing.target = undefined;
   owner.homingReady = false;
   owner.turret.gotoAndStop(18);
   owner.hitPointsFront = new Array();
   owner.hitPointsFront[0] = {x:(- owner.base._width) / 2,y:(- owner.base._height) / 2};
   owner.hitPointsFront[1] = {x:(- owner.base._width) / 4,y:(- owner.base._height) / 2};
   owner.hitPointsFront[2] = {x:owner.base._width / 4,y:(- owner.base._height) / 2};
   owner.hitPointsFront[3] = {x:owner.base._width / 2,y:(- owner.base._height) / 2};
   owner.hitPointsFront[4] = {x:0,y:(- owner.base._height) / 2};
   var _loc6_ = new Color(owner.turret.background);
   _loc6_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   _loc6_ = new Color(owner.scoreboard.tankIcon.turretBackground);
   _loc6_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   _loc6_ = new Color(owner.scoreboard.tankIcon.turretBackground2);
   _loc6_.setTint(owner.turretColor.r,owner.turretColor.g,owner.turretColor.b,owner.turretColor.a);
   var _loc5_ = 0;
   while(_loc5_ < 3 * _root.HOMINGSMOKECLOUDS)
   {
      var _loc4_ = _root.game.getNextHighestDepth();
      _root.game.createEmptyMovieClip("homingSmoke-" + _loc4_,_root.game.getNextHighestDepth());
      s = _root.game["homingSmoke-" + _loc4_];
      s.lineStyle(8 * (_root.SCALE / 50),Math.round(random(4) + 6) * 1118481,40);
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = (2 * Math.random() - 1) * (_root.SCALE / 50) - Math.random() * Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 0.1;
      s.yspeed = (2 * Math.random() - 1) * (_root.SCALE / 50) - Math.random() * Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 0.1;
      s.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 2 / 16;
      s.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 2 / 16;
      s._x = s.x;
      s._y = s.y;
      s.onEnterFrame = function()
      {
         if(_root.frozen)
         {
            return undefined;
         }
         this._xscale += 2;
         this._yscale += 2;
         this._alpha -= 4 - Math.random() * 3;
         this.xspeed *= 0.9000000000000002;
         this.yspeed *= 0.9000000000000002;
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc5_ = _loc5_ + 1;
   }
}
function layMine(owner)
{
   if(_root.soundOn)
   {
      _root.soundBullet.start();
   }
   mineDepth = _root.game.mazebg.getNextHighestDepth();
   mineName = "mine-" + mineDepth;
   mine = _root.game.mazebg.attachMovie("mine",mineName,mineDepth);
   mine.x = owner._x;
   mine.y = owner._y;
   mine._x = mine.x;
   mine._y = mine.y;
   mine._xscale = 100 * (_root.SCALE / 50);
   mine._yscale = 100 * (_root.SCALE / 50);
   mine.xSpeed = (- Math.cos((owner._rotation - 90) * 3.141592653589793 / 180)) * MINESPEED / MINEHITCHECKINTERVALS * (_root.SCALE / 50);
   mine.ySpeed = (- Math.sin((owner._rotation - 90) * 3.141592653589793 / 180)) * MINESPEED / MINEHITCHECKINTERVALS * (_root.SCALE / 50);
   mine.deadly = MINEDEADLY;
   mine.hideCounter = MINEHIDETIME;
   mine.landed = false;
   mine.armed = false;
   mine.detonating = false;
   mine.detonateCounter = MINEDETONATETIME;
   mine.owner = owner;
   owner.minesLayed = owner.minesLayed + 1;
   if(owner.minesLayed >= MINENUMBER)
   {
      _root.setWeapon(owner,"bullet");
   }
}
function fireRemote(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundBullet.start();
   }
   remoteDepth = _root.game.getNextHighestDepth();
   remoteName = "remote" + remoteDepth;
   remote = _root.game.attachMovie("remotebullet",remoteName,remoteDepth);
   owner.swapDepths(remote);
   remote.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   remote.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   remote._x = remote.x;
   remote._y = remote.y;
   remote._rotation = owner._rotation;
   remote.turnSpeed = REMOTETURNSPEED;
   remote._xscale = 100 * (_root.SCALE / 50);
   remote._yscale = 100 * (_root.SCALE / 50);
   remote.xSpeed = Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * REMOTESPEED / REMOTEHITCHECKINTERVALS * (_root.SCALE / 50);
   remote.ySpeed = Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * REMOTESPEED / REMOTEHITCHECKINTERVALS * (_root.SCALE / 50);
   remote.lifetime = REMOTELIFETIME;
   remote.deadly = REMOTEDEADLY;
   remote.owner = owner;
   owner.remoteControlling = true;
}
function fireElectric(owner)
{
   owner.turret.play();
   if(_root.soundOn)
   {
      _root.soundBullet.start();
   }
   bulletDepth = _root.game.getNextHighestDepth();
   bulletName = "electricbullet" + bulletDepth;
   bullet = _root.game.attachMovie("electricbullet",bulletName,bulletDepth);
   owner.swapDepths(bullet);
   bullet.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet._x = bullet.x;
   bullet._y = bullet.y;
   bullet._xscale = 100 * (_root.SCALE / 50);
   bullet._yscale = 100 * (_root.SCALE / 50);
   bullet.xSpeed = Math.cos((owner._rotation - 95) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.ySpeed = Math.sin((owner._rotation - 95) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet.lifetime = BULLETLIFETIME;
   bullet.deadly = BULLETDEADLY;
   bullet.owner = owner;
   bullet2Depth = _root.game.getNextHighestDepth();
   bullet2Name = "electricbullet" + bullet2Depth;
   bullet2 = _root.game.attachMovie("electricbullet",bullet2Name,bullet2Depth);
   owner.swapDepths(bullet2);
   bullet2.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet2.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * SCALE * 4.5 / 16;
   bullet2._x = bullet2.x;
   bullet2._y = bullet2.y;
   bullet2._xscale = 100 * (_root.SCALE / 50);
   bullet2._yscale = 100 * (_root.SCALE / 50);
   bullet2.xSpeed = Math.cos((owner._rotation - 85) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet2.ySpeed = Math.sin((owner._rotation - 85) * 3.141592653589793 / 180) * BULLETSPEED / BULLETHITCHECKINTERVALS * (_root.SCALE / 50);
   bullet2.lifetime = BULLETLIFETIME;
   bullet2.deadly = BULLETDEADLY;
   bullet2.owner = owner;
   bullet.firstBullet = bullet;
   bullet.lastBullet = bullet2;
   bullet2.firstBullet = bullet;
   bullet2.lastBullet = bullet2;
   owner.electricReady = false;
   sparkDepth = _root.game.getNextHighestDepth();
   sparkName = "spark" + sparkDepth;
   spark = _root.game.createEmptyMovieClip(sparkName,sparkDepth);
   spark._x = 0;
   spark._y = 0;
   spark.points = new Array();
   spark.waveCounter = 0;
   bullet.sparkMC = spark;
   bullet2.sparkMC = spark;
}
function placeCrate(pos, scale)
{
   if(_root.settingsActiveWeapons.length == 0)
   {
      return undefined;
   }
   _root.numberOfCrates = _root.numberOfCrates + 1;
   if(_root.soundOn)
   {
      _root.soundCrate.start();
   }
   var _loc4_ = _root.game.mazebg.attachMovie("crate","crate" + _root.reachable[pos].x + "-" + _root.reachable[pos].y,_root.game.mazebg.getNextHighestDepth());
   _loc4_._x = (_root.reachable[pos].x + 0.5) * scale;
   _loc4_._y = (_root.reachable[pos].y + 0.5) * scale;
   _loc4_.targetScale = scale * 1.5;
   _loc4_._rotation = Math.random() * 90 - 45;
   _loc4_._xscale = 0;
   _loc4_._yscale = 0;
   _loc4_.scaleSpeed = scale / 5;
   _loc4_.scaleSpeedDiff = scale / 20;
   _loc4_.landed = false;
   var _loc3_ = 0;
   while(_loc3_ < _root.NUMBEROFPUFFCLOUDS)
   {
      _root.game.mazebg.createEmptyMovieClip("puff" + _root.numberOfCrates + "-" + _loc3_,_root.game.mazebg.getNextHighestDepth());
      s = _root.game.mazebg["puff" + _root.numberOfCrates + "-" + _loc3_];
      s.lineStyle(15 * (_root.SCALE / 50),8947848 + Math.random() * 16777215,40 + random(20));
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.yspeed = (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.x = _loc4_._x + s.xspeed * (Math.random() * 3 + 1) + (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s.y = _loc4_._y + s.yspeed * (Math.random() * 3 + 1) + (Math.random() * 2 - 1) * (_root.SCALE / 50);
      s._x = s.x;
      s._y = s.y;
      s.onEnterFrame = function()
      {
         if(_root.frozen)
         {
            return undefined;
         }
         this._xscale += 2;
         this._yscale += 2;
         this._alpha -= 8 - Math.random() * 5;
         this.xspeed *= 0.85;
         this.yspeed *= 0.85;
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc3_ = _loc3_ + 1;
   }
   weapon = random(_root.settingsActiveWeapons.length);
   _loc4_.gotoAndStop(_root.settingsActiveWeapons[weapon]);
   _loc4_.weapon = _root.settingsActiveWeapons[weapon];
   _loc4_.pos = pos;
   _root.reachable[pos].used = true;
}
function addAimer(owner)
{
   aimerDepth = _root.game.getNextHighestDepth();
   aimerName = "aimer" + aimerDepth;
   aimer = _root.game.attachMovie("aimer",aimerName,aimerDepth);
   owner.swapDepths(aimer);
   aimer.owner = owner;
   aimer.aimerColor = 16711680 * owner.turretColor.r / 255 + 65280 * owner.turretColor.g / 255 + 255 * owner.turretColor.b / 255;
   return aimer;
}
function addShield(owner)
{
   shieldDepth = _root.game.getNextHighestDepth();
   shieldName = "shield" + shieldDepth;
   shield = _root.game.attachMovie("shield",shieldName,shieldDepth);
   shield.owner = owner;
   shield.shieldColor = 16711680 * owner.baseColor.r / 255 + 65280 * owner.baseColor.g / 255 + 255 * owner.baseColor.b / 255;
   shield.targetSize = SHIELDSIZE;
   return shield;
}
function setupBattle()
{
   _root.player1ScoreBoard.score.text = _root.player1ScoreBoard.score.text;
   _root.player2ScoreBoard.score.text = _root.player2ScoreBoard.score.text;
   _root.player3ScoreBoard.score.text = _root.player3ScoreBoard.score.text;
   if(_root.TANKS == 2)
   {
      _root.player1ScoreBoard.score.autoSize = "left";
      _root.player1ScoreBoard.score._x = _root.player1ScoreBoard.tankIcon._width / 2 - 5;
      _root.player1ScoreBoard._x = MOVIEWIDTH / 2 - 150;
      _root.player1ScoreBoard._y = MOVIEHEIGHT - 25;
      if(_root.loginInfo.actualRankedPlayers[0])
      {
         _root.player1ScoreBoard.username.autoSize = "center";
         _root.player1ScoreBoard.username.text = _root.loginInfo.p1n;
      }
      else
      {
         _root.player1ScoreBoard.username.text = "";
      }
      _root.player2ScoreBoard.score.autoSize = "right";
      _root.player2ScoreBoard.tankIcon._xscale = -100;
      if(firstSetup)
      {
         _root.player2ScoreBoard.score._x = (- _root.player2ScoreBoard.tankIcon._width) / 2 - 18.5 - _root.player2ScoreBoard.score._width;
      }
      _root.player2ScoreBoard._x = MOVIEWIDTH / 2 + 190;
      _root.player2ScoreBoard._y = MOVIEHEIGHT - 25;
      if(_root.loginInfo.actualRankedPlayers[1])
      {
         _root.player2ScoreBoard.username.autoSize = "center";
         _root.player2ScoreBoard.username.text = _root.loginInfo.p2n;
      }
      else
      {
         _root.player2ScoreBoard.username.text = "";
      }
      _root.player3ScoreBoard._visible = false;
   }
   else if(_root.TANKS == 3)
   {
      _root.player1ScoreBoard.score.autoSize = "left";
      _root.player1ScoreBoard.score._x = _root.player1ScoreBoard.tankIcon._width / 2 - 5;
      _root.player1ScoreBoard._x = MOVIEWIDTH / 2 - 180;
      _root.player1ScoreBoard._y = MOVIEHEIGHT - 25;
      if(_root.loginInfo.actualRankedPlayers[0])
      {
         _root.player1ScoreBoard.username.autoSize = "center";
         _root.player1ScoreBoard.username.text = _root.loginInfo.p1n;
      }
      else
      {
         _root.player1ScoreBoard.username.text = "";
      }
      _root.player2ScoreBoard.score.autoSize = "left";
      _root.player2ScoreBoard.score._x = _root.player2ScoreBoard.tankIcon._width / 2 - 5;
      _root.player2ScoreBoard._x = MOVIEWIDTH / 2;
      _root.player2ScoreBoard._y = MOVIEHEIGHT - 25;
      if(_root.loginInfo.actualRankedPlayers[1])
      {
         _root.player2ScoreBoard.username.autoSize = "center";
         _root.player2ScoreBoard.username.text = _root.loginInfo.p2n;
      }
      else
      {
         _root.player2ScoreBoard.username.text = "";
      }
      _root.player3ScoreBoard.score.autoSize = "right";
      _root.player3ScoreBoard.tankIcon._xscale = -100;
      if(firstSetup)
      {
         _root.player3ScoreBoard.score._x = (- _root.player3ScoreBoard.tankIcon._width) / 2 - 18.5 - _root.player3ScoreBoard.score._width;
      }
      _root.player3ScoreBoard._x = MOVIEWIDTH / 2 + 220;
      _root.player3ScoreBoard._y = MOVIEHEIGHT - 25;
      if(_root.loginInfo.actualRankedPlayers[2])
      {
         _root.player3ScoreBoard.username.autoSize = "center";
         _root.player3ScoreBoard.username.text = _root.loginInfo.p3n;
      }
      else
      {
         _root.player3ScoreBoard.username.text = "";
      }
   }
   if(_root.settingsPanelRequested)
   {
      if(firstSetup)
      {
         var _loc3_ = 0;
         while(_loc3_ < TANKS)
         {
            var _loc15_ = undefined;
            var _loc5_ = _root["player" + (_loc3_ + 1) + "ScoreBoard"].tankIcon;
            _loc5_.gotoAndStop(1);
            var _loc4_ = undefined;
            var _loc2_ = undefined;
            var _loc10_ = parseInt(_root.loginInfo["p" + (_loc3_ + 1) + "bc"]);
            var _loc9_ = parseInt(_root.loginInfo["p" + (_loc3_ + 1) + "tc"]);
            _loc4_ = convertFromHexToRGB(_loc10_);
            _loc2_ = convertFromHexToRGB(_loc9_);
            var _loc8_ = new Color(_loc5_.turretBackground);
            _loc8_.setTint(_loc2_.r,_loc2_.g,_loc2_.b,_loc2_.a);
            _loc8_ = new Color(_loc5_.turretBackground2);
            _loc8_.setTint(_loc2_.r,_loc2_.g,_loc2_.b,_loc2_.a);
            var _loc7_ = new Color(_loc5_.tracks.background);
            _loc7_.setTint(_loc4_.r,_loc4_.g,_loc4_.b,_loc4_.a);
            _loc3_ = _loc3_ + 1;
         }
      }
      openSettingsPanel();
      aliveCount = TANKS;
      return undefined;
   }
   _root.createEmptyMovieClip("game",0);
   var _loc6_ = new Array(TANKS);
   _root.reachable = new Array();
   var _loc13_ = new Array();
   if(_root.settingsPlayRandomMazes)
   {
      _loc13_.push({f:setupStandardMaze,arg1:_loc6_});
   }
   if(_root.settingsPlayMyCustomMazes)
   {
      _loc13_.push({f:setupCustomMaze,arg1:myMazesFetcher,arg2:_loc6_});
   }
   if(_root.settingsPlayOtherCustomMazes)
   {
      _loc13_.push({f:setupCustomMaze,arg1:otherPeoplesMazesFetcher,arg2:_loc6_});
   }
   var _loc17_ = Math.floor(Math.random() * _loc13_.length);
   if(_loc13_.length == 0)
   {
      setupStandardMaze(_loc6_);
   }
   else
   {
      _loc13_[_loc17_].f(_loc13_[_loc17_].arg1,_loc13_[_loc17_].arg2);
   }
   _loc3_ = 0;
   while(_loc3_ < TANKS)
   {
      deployTank(_loc6_[_loc3_],_loc3_,SCALE);
      _loc3_ = _loc3_ + 1;
   }
   var _loc20_ = Math.floor(maze.length * SCALE);
   var _loc21_ = Math.floor(maze[0].length * SCALE);
   var _loc14_ = Math.floor(SCALE / 16);
   var _loc16_ = 1;
   var _loc18_ = _loc20_ + 2 * _loc14_ + 2 * _loc16_;
   var _loc19_ = _loc21_ + 2 * _loc14_ + 2 * _loc16_;
   if(_loc18_ > MOVIEWIDTH)
   {
      _loc18_ = MOVIEWIDTH;
   }
   if(_loc19_ > MOVIEHEIGHT - HEIGHTTOBOTTOM)
   {
      _loc19_ = MOVIEHEIGHT - HEIGHTTOBOTTOM;
   }
   gameX = 10 + _loc14_ + _loc16_ + Math.floor(MOVIEWIDTH / 2 - _loc18_ / 2);
   gameY = 10 + _loc14_ + _loc16_;
   _root.game._x = gameX;
   _root.game._y = gameY;
   aliveCount = TANKS;
   firstSetup = false;
   distancesForMaze = new Array(maze.length);
   _loc3_ = 0;
   while(_loc3_ < distancesForMaze.length)
   {
      distancesForMaze[_loc3_] = new Array(maze[_loc3_].length);
      _loc3_ = _loc3_ + 1;
   }
   _loc3_ = 0;
   while(_loc3_ < reachable.length)
   {
      distancesForMaze[reachable[_loc3_].x][reachable[_loc3_].y] = calcDistances(maze,reachable[_loc3_].x,reachable[_loc3_].y);
      _loc3_ = _loc3_ + 1;
   }
   tankFields = new Array(_root.TANKS);
   _loc3_ = 0;
   while(_loc3_ < _root.TANKS)
   {
      tankFields[_loc3_] = {x:_loc6_[_loc3_].x,y:_loc6_[_loc3_].y};
      _loc3_ = _loc3_ + 1;
   }
   deadEnds = findDeadEnds(maze,reachable);
}
function setupStandardMaze(tanks)
{
   while(_root.reachable.length < 2 * TANKS)
   {
      WIDTH = Math.floor(random(9)) + 4;
      HEIGHT = Math.floor(random(7)) + 4;
      SCALE = Math.min((MOVIEHEIGHT - HEIGHTTOBOTTOM) / (HEIGHT + 0.125),MOVIEWIDTH / (WIDTH + 0.125));
      maze = createMaze(WIDTH,HEIGHT);
      tanks[0] = {x:Math.floor(Math.random() * WIDTH),y:Math.floor(Math.random() * HEIGHT)};
      _root.reachable = calcReachable(maze,tanks[0].x,tanks[0].y);
   }
   _root.reachable[0].used = true;
   var _loc3_ = 1;
   while(_loc3_ < TANKS)
   {
      var _loc2_ = Math.floor(Math.random() * _root.reachable.length);
      if(_root.reachable[_loc2_].used != true)
      {
         tanks[_loc3_] = {x:_root.reachable[_loc2_].x,y:_root.reachable[_loc2_].y};
         _root.reachable[_loc2_].used = true;
      }
      else
      {
         _loc3_ = _loc3_ - 1;
      }
      _loc3_ = _loc3_ + 1;
   }
   _loc3_ = 0;
   while(_loc3_ < _root.reachable.length)
   {
      _root.reachable[_loc3_].used = false;
      _loc3_ = _loc3_ + 1;
   }
   _root.numberOfCrates = 0;
   drawMaze(maze,SCALE);
}
function setupCustomMaze(fetcher, tanks)
{
   maze = fetcher.createMaze();
   if(maze == undefined)
   {
      setupStandardMaze(tanks);
      if(!_root.settingsPlayRandomMazes)
      {
         displayErrorSign();
      }
      return undefined;
   }
   WIDTH = maze.length;
   HEIGHT = maze[0].length;
   SCALE = Math.min((MOVIEHEIGHT - HEIGHTTOBOTTOM) / (HEIGHT + 0.125),MOVIEWIDTH / (WIDTH + 0.125));
   var _loc3_ = fetcher.getSpawnPoints();
   var _loc11_ = fetcher.getGrounds();
   var _loc7_ = 0;
   while(_loc3_.length > 0)
   {
      var _loc4_ = Math.floor(Math.random() * _loc3_.length);
      tanks[_loc7_] = {x:_loc3_[_loc4_].x,y:_loc3_[_loc4_].y};
      _loc7_ = _loc7_ + 1;
      _loc3_.splice(_loc4_,1);
   }
   if(_loc7_ == 0)
   {
      var _loc2_ = Math.floor(Math.random() * _loc11_.length);
      tanks[0] = {x:_loc11_[_loc2_].x,y:_loc11_[_loc2_].y};
      _loc7_ = _loc7_ + 1;
   }
   _root.reachable = calcReachable(maze,tanks[0].x,tanks[0].y);
   _root.reachable[0].used = true;
   var _loc5_ = 1;
   while(_loc5_ < _loc7_)
   {
      _root.reachable[_root.reachableIndex[tanks[_loc5_].x][tanks[_loc5_].y]].used = true;
      _loc5_ = _loc5_ + 1;
   }
   _loc5_ = _loc7_;
   while(_loc5_ < TANKS)
   {
      _loc2_ = Math.floor(Math.random() * _root.reachable.length);
      if(_root.reachable[_loc2_].used != true)
      {
         tanks[_loc5_] = {x:_root.reachable[_loc2_].x,y:_root.reachable[_loc2_].y};
         _root.reachable[_loc2_].used = true;
      }
      else
      {
         _loc5_ = _loc5_ - 1;
      }
      _loc5_ = _loc5_ + 1;
   }
   _root.crateSpawnPoints = fetcher.getCrateSpawnPoints();
   _loc5_ = 0;
   while(_loc5_ < _root.reachable.length)
   {
      _root.reachable[_loc5_].used = false;
      _loc5_ = _loc5_ + 1;
   }
   _root.numberOfCrates = 0;
   drawMaze(maze,SCALE);
   mazeTitle.text = fetcher.getTitle();
   mazeCreatedBy.text = "by " + fetcher.getCreator();
   mazeTitle._visible = true;
   mazeCreatedBy._visible = true;
}
function cleanUpBattle()
{
   _root.game.removeMovieClip();
   _root.scopeCross.removeMovieClip();
   _root.scopeCircle.removeMovieClip();
   mazeTitle._visible = false;
   mazeCreatedBy._visible = false;
}
function registerHit(owner, victim)
{
   if(_root.AIEnabled && victim.username == _root.AIName)
   {
      _root.laika.growl();
   }
   else if(_root.AIEnabled && victim.username != _root.AIName)
   {
      _root.laika.scoff();
   }
   if(_root.loginInfo.rankedMatch)
   {
      var _loc3_ = _root.attachMovie("serverInfo","serverInfo" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
      _loc3_.killer = owner;
      _loc3_.victim = victim;
      _loc3_._x = 30;
      _loc3_._y = 30;
      _loc3_._alpha = -30;
      _loc3_.balls._rotation = _root.serverInfoRotation;
      _loc3_.variablesLoaded = false;
      _loc3_.loadVariablesString = "includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("tankScrapped=" + (!_root.AIEnabled ? TANKS : 1) + "&x=" + _root.loginInfo.x + "&victim=" + victim.username + "&killer=" + owner.username + "&what=" + "" + "&how=" + "" + "&when=" + ""));
      _loc3_.onEnterFrame = function()
      {
         this.balls._rotation = _root.serverInfoRotation;
         if(this.response != undefined)
         {
            this._alpha -= 10;
            if(this._alpha <= 0)
            {
               this.removeMovieClip();
            }
         }
         else if(this._alpha < 100)
         {
            this._alpha += 10;
         }
         if(this.r != undefined)
         {
            this.response = _root.decodeMessage(this.r);
            if(this.response.error == undefined)
            {
               _root.scrapyard = this.response.scrapyard;
               var _loc3_ = "";
               if(this.killer != this.victim)
               {
                  _loc3_ += "javascript: updateScore(\'" + this.killer.username.toLowerCase() + "-kills\', 1);";
               }
               _loc3_ += "javascript: updateScore(\'" + this.victim.username.toLowerCase() + "-deaths\', 1); void ( 0 );";
               getURL(_loc3_,"");
            }
            else
            {
               displayErrorSign();
            }
            serverInfoQueue.shift();
            this.r = undefined;
         }
      };
      serverInfoQueue.push(_loc3_);
   }
   else
   {
      _root.loadVariables("includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("tankScrapped=" + _root.TANKS + "&a=" + Math.random() + "&b=" + Math.random())),"POST");
   }
}
function assignPoints()
{
   var _loc5_ = 0;
   var _loc4_ = 0;
   var _loc2_ = 0;
   while(_loc2_ < _root.TANKS)
   {
      if(!_root.game["tank" + _loc2_].alive)
      {
         _loc5_ += Math.max(1,_root.loginInfo["p" + (_loc2_ + 1) + "e"]);
      }
      else
      {
         _loc4_ = Math.max(1,_root.loginInfo["p" + (_loc2_ + 1) + "e"]);
      }
      _loc2_ = _loc2_ + 1;
   }
   var _loc3_ = Math.round(Math.min(20,Math.max(1,10 + (_loc5_ - _loc4_) / 100)));
   _loc2_ = 0;
   while(_loc2_ < _root.TANKS)
   {
      if(_root.game["tank" + _loc2_].alive)
      {
         _root.game["tank" + _loc2_].scoreboard.add(1);
         if(_root.loginInfo.actualRankedPlayers[_loc2_])
         {
            getURL("javascript: updateMultipleScores(\'" + _root.loginInfo["p" + (_loc2_ + 1) + "n"].toLowerCase() + "-score\', 1, \'" + _root.loginInfo["p" + (_loc2_ + 1) + "n"].toLowerCase() + "-experience\', " + _loc3_ + "); void ( 0 );","");
            if(_root.loginInfo["p" + (_loc2_ + 1) + "n"] != "Laika")
            {
               _root.loginInfo["p" + (_loc2_ + 1) + "e"] = Number(_root.loginInfo["p" + (_loc2_ + 1) + "e"]) + _loc3_;
            }
         }
      }
      _loc2_ = _loc2_ + 1;
   }
}
function openSettingsPanel()
{
   _root.settingsPanelRequested = false;
   _root.frozen = true;
   _root.settings.gotoAndStop(1 + _root.settingsPanelRequested);
   var _loc4_ = _root.attachMovie("settingsPanel","settingsPanel" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
   _loc4_._x = Stage.width / 2;
   _loc4_._y = Stage.height / 2;
   _loc4_._xscale = 0;
   _loc4_._yscale = 0;
   var _loc2_ = 0;
   while(_loc2_ < _root.settingsActiveWeapons.length)
   {
      var _loc3_ = _root.settingsActiveWeapons[_loc2_];
      _loc4_[_loc3_ + "Slider"].activate = true;
      _loc2_ = _loc2_ + 1;
   }
   _loc4_.randomMazeSlider.activate = _root.settingsPlayRandomMazes;
   _loc4_.myOwnMazeSlider.activate = _root.settingsPlayMyCustomMazes;
   _loc4_.othersMazeSlider.activate = _root.settingsPlayOtherCustomMazes;
   _loc4_.newMouseControlSlider.activate = _root.settingsUseNewMouseControl;
   _loc4_.wasMouseShowing = Mouse.show();
   if(!_loc4_.wasMouseShowing)
   {
      _root.scopeCross._alpha = 0;
      _root.scopeCircle._alpha = 0;
   }
}
function displayErrorSign()
{
   var _loc3_ = _root.attachMovie("serverInfo","serverInfo" + _root.getNextHighestDepth(),_root.getNextHighestDepth(),{lifetime:25});
   _loc3_._x = 30;
   _loc3_._y = 30;
   _loc3_._alpha = -100;
   _loc3_.gotoAndStop(2);
   _loc3_.onEnterFrame = function()
   {
      if(this._alpha >= 100)
      {
         this.lifetime = this.lifetime - 1;
      }
      if(this.lifetime == 0)
      {
         this._alpha -= 10;
      }
      else if(this._alpha < 100)
      {
         this._alpha += 10;
      }
      if(this.lifetime == 0 && this._alpha <= 0)
      {
         this.removeMovieClip();
      }
   };
}
function recordFrame()
{
   var _loc4_ = {};
   var _loc2_ = 0;
   while(_loc2_ < _root.TANKS)
   {
      if(_root.game["tank" + _loc2_].alive)
      {
         _loc4_["tank" + _loc2_] = {};
         _loc4_["tank" + _loc2_]._x = _root.game["tank" + _loc2_]._x;
         _loc4_["tank" + _loc2_]._y = _root.game["tank" + _loc2_]._y;
         _loc4_["tank" + _loc2_]._rotation = _root.game["tank" + _loc2_]._rotation;
      }
      _loc2_ = _loc2_ + 1;
   }
   var _loc5_ = new Array();
   for(var _loc6_ in _root.game)
   {
      if(_loc6_.substr(0,6) == "bullet")
      {
         var _loc3_ = {};
         _loc3_._x = _root.game[_loc6_]._x;
         _loc3_._y = _root.game[_loc6_]._y;
         _loc5_.push(_loc3_);
      }
   }
   _loc4_.bullets = _loc5_;
   return _loc4_;
}
function playFrame(frame)
{
   trace(frame.tank0);
   for(var _loc3_ in frame)
   {
      trace(_loc3_);
      if(_loc3_.substr(0,4) == "tank")
      {
         trace(_root.game[_loc3_]);
         _root.game[_loc3_].x = frame[_loc3_]._x;
         _root.game[_loc3_].y = frame[_loc3_]._y;
         _root.game[_loc3_].oldRot = frame[_loc3_]._rotation;
         _root.game[_loc3_]._rotation = frame[_loc3_]._rotation;
      }
   }
}
stop();
MOVIEWIDTH = 692;
MOVIEHEIGHT = 480;
HEIGHTTOBOTTOM = 80;
STARTWEAPON = "bullet";
BULLETSPEED = 4.5;
BULLETLIFETIME = 250;
LASERSPEED = 70;
LASERLIFETIME = 8;
AIMERLENGTH = 200;
FRAGSPEED = 4.5;
FRAGLIFETIME = 250;
DEATHRAYLIFETIME = 30;
HOMINGSPEED = 4.5;
HOMINGLIFETIME = 250;
MINESPEED = 5.5;
REMOTESPEED = 4.5;
REMOTETURNSPEED = 15;
REMOTELIFETIME = 250;
GATLINGSPEED = 5.5;
GATLINGLIFETIME = 125;
GATLINGBULLETS = 20;
CRATESPAWNTIMEBASE = 350;
CRATESPAWNTIMERANDOM = 200;
CRATESPAWNMAZESIZESCALE = 2000;
BULLETHITCHECKINTERVALS = 7;
LASERHITCHECKINTERVALS = 70;
AIMERHITCHECKINTERVALS = 100;
FRAGHITCHECKINTERVALS = 7;
HOMINGHITCHECKINTERVALS = 5;
MINEHITCHECKINTERVALS = 4;
REMOTEHITCHECKINTERVALS = 5;
GATLINGHITCHECKINTERVALS = 7;
BULLETDEADLY = 0;
LASERDEADLY = 7;
AIMERACTIVE = 7;
FRAGDEADLY = 0;
DEATHRAYDEADLY = 0;
HOMINGDEADLY = 0;
MINEDEADLY = 20;
REMOTEDEADLY = 0;
GATLINGDEADLY = 2;
NUMBEROFFRAGMENTS = 8;
NUMBEROFSMOKECLOUDS = 10;
NUMBEROFPUFFCLOUDS = 0;
NUMBEROFDUSTCLOUDS = 20;
NUMBEROFFRAMESBEFOREEND = 125;
NUMBEROFFRAMESFROZEN = 50;
NUMBEROFFRAMESBEFORERESET = 5;
FRAGFRAGMENTS = 40;
FRAGSMOKECLOUDS = 10;
FRAGLEVELS = 1;
GATLINGSPINSPEED = 10;
DEATHRAYWARMUPTIME = 9;
HOMINGSTARTUPTIME = 50;
HOMINGSMOKECLOUDS = 5;
MINEHIDETIME = 0;
MINENUMBER = 3;
MINEFRAGMENTS = 40;
MINESMOKECLOUDS = 10;
MINEDETONATETIME = 6;
SHIELDSIZE = 70;
MAXSHAKE = 8;
aliveCount = 0;
endCount = -1;
resetCount = -1;
frozen = false;
shake = 0;
gameX = gameY = 0;
MAXDEADENDPENALTY = 5;
crateTimer = (CRATESPAWNTIMEBASE + random(CRATESPAWNTIMERANDOM)) * _root.settingsCrateSpawnModifier;
var outgoingLc = new LocalConnection();
var firstSetup = true;
var distancesForMaze;
var tankFields;
mazeTitle._visible = false;
mazeCreatedBy._visible = false;
if(_root.AIEnabled)
{
   settingsActiveWeapons = new Array();
}
setupBattle();
if(_root.AIEnabled)
{
   var laika = _root.attachMovie("Laika","Laika",_root.getNextHighestDepth());
   _root.player2ScoreBoard.swapDepths(_root.getNextHighestDepth());
   trace(_root.player2ScoreBoard.getDepth());
   laika._x = 530;
   laika._y = 424;
   laika._xscale = -40;
   laika._yscale = 40;
   laika.eyeAnimation = 0;
   laika.headRotationAnimation = 0;
   laika.headBobbingAnimation = 0;
   laika.headY = laika.head._y;
   laika.neckAY = laika.neckA._y;
   laika.neckBY = laika.neckB._y;
   laika.cylinderY = laika.cylinder._y;
   laika.targetHeadRotationAnimation = 0;
   laika.targetHeadBobbingAnimation = 0;
   laika.animationCounter = 0;
   laika.animation = "idle";
   laika.onEnterFrame = function()
   {
      switch(this.animation)
      {
         case "idle":
            this.targetHeadRotationAnimation = 0;
            this.targetHeadBobbingAnimation = this.animationCounter;
            this.head.gotoAndStop(1);
            this.headOutline.gotoAndStop(1);
            break;
         case "growl":
            this.targetHeadRotationAnimation = 1.5707963267948966 + Math.random() * 1;
            this.targetHeadBobbingAnimation = 1.5707963267948966 + Math.random() * 1;
            this.head.gotoAndStop(3);
            this.headOutline.gotoAndStop(3);
            if(this.animationCounter > 4)
            {
               this.idle();
            }
            break;
         case "scoff":
            this.targetHeadRotationAnimation = 1.5707963267948966;
            this.targetHeadBobbingAnimation = 1.5707963267948966 + this.animationCounter * 10;
            this.head.gotoAndStop(2);
            this.headOutline.gotoAndStop(2);
            if(this.animationCounter > 3)
            {
               this.idle();
               break;
            }
      }
      this.animationCounter += 0.1;
      if(random(30) == 0)
      {
         this.eyeAnimation = random(3);
      }
      if(this.eyeAnimation > 0)
      {
         this.eyeAnimation = this.eyeAnimation - 1;
         this.head.eye._visible = false;
      }
      else
      {
         this.head.eye._visible = true;
      }
      this.headRotationAnimation += (this.targetHeadRotationAnimation - this.headRotationAnimation) * 0.3;
      this.headBobbingAnimation += (this.targetHeadBobbingAnimation - this.headBobbingAnimation) * 0.3;
      t = Math.sin(this.headRotationAnimation);
      t2 = Math.sin(this.headBobbingAnimation);
      this.head._rotation = 10 * t;
      this.head._y = this.headY + 2 * t2;
      this.headOutline._rotation = this.head._rotation;
      this.headOutline._y = this.head._y;
      this.neckA._rotation = 4 * t;
      this.neckA._y = this.neckAY + 3 * t2;
      this.neckAOutline._rotation = this.neckA._rotation;
      this.neckAOutline._y = this.neckA._y;
      this.neckB._rotation = -2 * t;
      this.neckB._y = this.neckBY + 4 * t2;
      this.neckBOutline._rotation = this.neckB._rotation;
      this.neckBOutline._y = this.neckB._y;
      this.cylinder._rotation = 3 * t;
      this.cylinder._y = this.cylinderY + 4 * t2;
   };
   laika.idle = function()
   {
      this.animation = "idle";
      while(this.targetHeadRotationAnimation > 6.283185307179586)
      {
         this.targetHeadRotationAnimation -= 6.283185307179586;
         this.headRotationAnimation -= 6.283185307179586;
      }
      while(this.targetHeadBobbingAnimation > 6.283185307179586)
      {
         this.targetHeadBobbingAnimation -= 6.283185307179586;
         this.headBobbingAnimation -= 6.283185307179586;
      }
      this.animationCounter = 0;
   };
   laika.growl = function()
   {
      this.animation = "growl";
      while(this.targetHeadRotationAnimation > 6.283185307179586)
      {
         this.targetHeadRotationAnimation -= 6.283185307179586;
         this.headRotationAnimation -= 6.283185307179586;
      }
      while(this.targetHeadBobbingAnimation > 6.283185307179586)
      {
         this.targetHeadBobbingAnimation -= 6.283185307179586;
         this.headBobbingAnimation -= 6.283185307179586;
      }
      this.animationCounter = 0;
   };
   laika.scoff = function()
   {
      this.animation = "scoff";
      while(this.targetHeadRotationAnimation > 6.283185307179586)
      {
         this.targetHeadRotationAnimation -= 6.283185307179586;
         this.headRotationAnimation -= 6.283185307179586;
      }
      while(this.targetHeadBobbingAnimation > 6.283185307179586)
      {
         this.targetHeadBobbingAnimation -= 6.283185307179586;
         this.headBobbingAnimation -= 6.283185307179586;
      }
      this.animationCounter = 0;
   };
}
serverInfoRotation = 0;
serverInfoQueue = new Array();
var frames = new Array();
_root.onEnterFrame = function()
{
   _root.serverInfoRotation += 4;
   if(_root.serverInfoRotation > 360)
   {
      _root.serverInfoRotation -= 360;
   }
   if(serverInfoQueue[0] != undefined && !serverInfoQueue[0].variablesLoaded)
   {
      serverInfoQueue[0].variablesLoaded = true;
      serverInfoQueue[0].loadVariables(serverInfoQueue[0].loadVariablesString,"POST");
   }
   if(_root.loginInfo == undefined)
   {
      if(_root.r != undefined)
      {
         _root.scrapyard = decodeMessage(_root.r).scrapyard;
      }
   }
   if(_root.scrapyard != undefined)
   {
      outgoingLc.send("lcStats","updateScrapyard",_root.scrapyard);
      _root.scrapyard = undefined;
   }
   var _loc3_ = 0;
   while(_loc3_ < _root.TANKS)
   {
      var _loc8_ = Math.floor(_root.game["tank" + _loc3_]._x / _root.SCALE);
      var _loc9_ = Math.floor(_root.game["tank" + _loc3_]._y / _root.SCALE);
      tankFields[_loc3_] = {x:_loc8_,y:_loc9_};
      _loc3_ = _loc3_ + 1;
   }
   if(!frozen)
   {
      crateTimer--;
   }
   if(!frozen && crateTimer <= 0)
   {
      crateTimer = (CRATESPAWNTIMEBASE + random(CRATESPAWNTIMERANDOM) + CRATESPAWNMAZESIZESCALE / _root.reachable.length) * _root.settingsCrateSpawnModifier;
      pos = 0;
      var _loc6_ = true;
      var _loc10_ = -1;
      var _loc14_ = false;
      if(_root.numberOfCrates >= _root.settingsMaxCrates)
      {
         _loc14_ = true;
      }
      while((_root.reachable[pos].used || _loc6_) && _loc10_ < 5 && !_loc14_)
      {
         if(_root.crateSpawnPoints.length > 0)
         {
            var _loc7_ = _root.crateSpawnPoints[random(_root.crateSpawnPoints.length)];
            pos = _root.reachableIndex[_loc7_.x][_loc7_.y];
         }
         else
         {
            pos = random(_root.reachable.length);
         }
         _loc6_ = false;
         _loc10_ = _loc10_ + 1;
         if(pos == undefined)
         {
            _loc6_ = true;
         }
         else
         {
            _loc3_ = 0;
            while(_loc3_ < _root.TANKS)
            {
               var _loc5_ = Math.floor(_root.game["tank" + _loc3_].x / SCALE);
               var _loc4_ = Math.floor(_root.game["tank" + _loc3_].y / SCALE);
               if(Math.abs(_root.reachable[pos].x - _loc5_) + Math.abs(_root.reachable[pos].y - _loc4_) <= 1 && _root.game["tank" + _loc3_].alive)
               {
                  _loc6_ = true;
               }
               _loc3_ = _loc3_ + 1;
            }
         }
      }
      if(_loc10_ < 5 && !_loc14_ && pos != undefined)
      {
         _root.placeCrate(pos,SCALE);
      }
   }
   if(shake >= 0)
   {
      _root.game._x = gameX + random(shake) - shake / 2;
      _root.game._y = gameY + random(shake) - shake / 2;
      shake -= 0.5;
   }
   if(aliveCount <= 1)
   {
      if(endCount >= 0)
      {
         endCount--;
      }
      if(endCount == NUMBEROFFRAMESFROZEN)
      {
         _root.game.tank0.turret.stop();
         _root.game.tank1.turret.stop();
         _root.game.tank2.turret.stop();
         frozen = true;
         if(_root.loginInfo.rankedMatch)
         {
            var _loc13_ = _root.attachMovie("serverInfo","serverInfo" + _root.getNextHighestDepth(),_root.getNextHighestDepth());
            _loc13_._x = 30;
            _loc13_._y = 30;
            _loc13_._alpha = -30;
            _loc13_.balls._rotation = _root.serverInfoRotation;
            _loc13_.variablesLoaded = false;
            _loc13_.loadVariablesString = "includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("roundCompleted=" + (!_root.AIEnabled ? TANKS : 1) + "&x=" + _root.loginInfo.x + "&a=" + Math.random() + "&b=" + Math.random()));
            _loc13_.onEnterFrame = function()
            {
               this.balls._rotation = _root.serverInfoRotation;
               if(this.response != undefined)
               {
                  this._alpha -= 10;
                  if(this._alpha <= 0)
                  {
                     this.removeMovieClip();
                  }
               }
               else if(this._alpha < 100)
               {
                  this._alpha += 10;
               }
               if(this.r != undefined)
               {
                  this.response = _root.decodeMessage(this.r);
                  if(this.response.error == undefined)
                  {
                     assignPoints();
                  }
                  else
                  {
                     displayErrorSign();
                  }
                  _root.loginInfo.x = this.response.x;
                  serverInfoQueue.shift();
                  this.r = undefined;
               }
               if(endCount <= 1)
               {
                  endCount++;
               }
            };
            serverInfoQueue.push(_loc13_);
         }
         else
         {
            _root.loadVariables("includes/updateGameStatistics.php?q=" + Base64.Encode(shuffleMessage("roundCompleted=" + TANKS + "&a=" + Math.random() + "&b=" + Math.random())),"POST");
            assignPoints();
         }
      }
      if(endCount == 0)
      {
         cleanUpBattle();
         resetCount = NUMBEROFFRAMESBEFORERESET;
      }
   }
   if(resetCount >= 0)
   {
      resetCount--;
   }
   if(resetCount == 0)
   {
      endCount = NUMBEROFFRAMESBEFOREEND + NUMBEROFFRAMESFROZEN;
      frozen = false;
      setupBattle();
   }
};
