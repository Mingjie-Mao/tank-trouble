function add(amount)
{
   var _loc5_ = Number(score.text);
   score.text = _loc5_ + amount;
   var _loc4_ = 0;
   while(_loc4_ < _root.NUMBEROFFRAGMENTS)
   {
      _root.createEmptyMovieClip("fragment" + number + "-" + _loc4_,_root.getNextHighestDepth());
      f = _root["fragment" + number + "-" + _loc4_];
      dir = 0.7853981633974483 + Math.random() * 3.141592653589793 * 0.5;
      speed = Math.random() * 3 + 1.5;
      f.xspeed = Math.cos(dir) * (speed / 1.5);
      f.yspeed = (- Math.sin(dir)) * (speed / 1.5);
      f.rotspeed = Math.random() * 120 - 60;
      f.active = true;
      f.smokenamebase = "smoke-fragment" + number + "-" + _loc4_;
      f.smokecounter = 0;
      f.hitPoints = new Array();
      f.lineStyle(2,3618615,100,false,"none");
      point1 = {x:random(10) - 5,y:random(10) - 5};
      point2 = {x:random(10) - 5,y:random(10) - 5};
      center = {x:(point1.x + point2.x) / 2,y:(point1.y + point2.y) / 2};
      f.moveTo(point1.x - center.x,point1.y - center.y);
      f.lineTo(point2.x - center.x,point2.y - center.y);
      f.hitPoints.push({x:point2.x - center.x,y:point2.y - center.y});
      f.hitPoints.push({x:point1.x - center.x,y:point1.y - center.y});
      f.spawnCounter = 0;
      f.x = this._x + score._x + score._width * Math.random();
      f.y = this._y + score._y + score._height * Math.random() * 0.5;
      f._x = f.x;
      f._y = f.y;
      f._rotation = random(360);
      f.onEnterFrame = function()
      {
         if(this.active)
         {
            this.x += this.xspeed;
            this.y += this.yspeed;
            this._x = this.x;
            this._y = this.y;
            this.xspeed *= 0.97;
            this.yspeed += 0.2;
            this.rotspeed *= 0.9900000000000002;
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
            if(_root.player1ScoreBoard._y + 13 < point.y)
            {
               return true;
            }
            _loc3_ = _loc3_ + 1;
         }
         return false;
      };
      _loc4_ = _loc4_ + 1;
   }
   _loc4_ = 0;
   while(_loc4_ < _root.NUMBEROFSMOKECLOUDS * 4)
   {
      _root.createEmptyMovieClip("smoke" + number + "-" + _loc4_,_root.getNextHighestDepth());
      s = _root["smoke" + number + "-" + _loc4_];
      s.lineStyle(8,Math.round(random(4) + 4) * 1118481,40 + random(20));
      s.moveTo(0,0);
      s.lineTo(0,1);
      s.xspeed = Math.random() * 1.5 - 0.75;
      s.yspeed = Math.random() * 1 - 2.5;
      s.x = this._x + score._x + score._width * Math.random();
      s.y = this._y + score._y + score._height * Math.random() * 0.75;
      s._x = s.x;
      s._y = s.y;
      s.onEnterFrame = function()
      {
         this._xscale += 2;
         this._yscale += 2;
         this._alpha -= 5 - Math.random() * 3;
         this.xspeed *= 0.93;
         this.yspeed *= 0.9500000000000001;
         this.x += this.xspeed;
         this.y += this.yspeed;
         this._x = this.x;
         this._y = this.y;
         if(this._alpha <= 0)
         {
            this.removeMovieClip();
         }
      };
      _loc4_ = _loc4_ + 1;
   }
}
function subtract(amount)
{
   var _loc1_ = Number(score.text);
   score.text = _loc1_ - amount;
}
