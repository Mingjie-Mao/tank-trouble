class MazeDataFetcher extends Object
{
   static var BUFFER_SIZE = 2;
   static var NUM_PLAYS = 3;
   function MazeDataFetcher(usernames)
   {
      super();
      this.usernames = usernames;
      this.userCount = 0;
      var _loc3_ = 0;
      while(_loc3_ < usernames.length)
      {
         if(usernames[_loc3_] != undefined)
         {
            this.userCount = this.userCount + 1;
         }
         _loc3_ = _loc3_ + 1;
      }
      this.data = new Array(MazeDataFetcher.BUFFER_SIZE);
      this.selectedMaze = 0;
      this.playCount = 0;
      this.spawnPoints = new Array();
      this.crateSpawnPoints = new Array();
      this.grounds = new Array();
      this.loaders = new Array(MazeDataFetcher.BUFFER_SIZE);
      _loc3_ = 0;
      while(_loc3_ < MazeDataFetcher.BUFFER_SIZE)
      {
         this.loaders[_loc3_] = new MazeDataLoader();
         _loc3_ = _loc3_ + 1;
      }
      _loc3_ = 0;
      while(_loc3_ < MazeDataFetcher.BUFFER_SIZE)
      {
         this.sendRequest(_loc3_);
         _loc3_ = _loc3_ + 1;
      }
   }
   function createMaze()
   {
      if(this.playCount == MazeDataFetcher.NUM_PLAYS)
      {
         this.finishWithMaze();
         this.selectedMaze = (this.selectedMaze + 1) % MazeDataFetcher.BUFFER_SIZE;
      }
      if(this.data[this.selectedMaze] == undefined)
      {
         this.sendRequest(this.selectedMaze);
         return undefined;
      }
      if(this.data[this.selectedMaze].notFound)
      {
         this.sendRequest(this.selectedMaze);
         return undefined;
      }
      var _loc11_ = this.data[this.selectedMaze].d.split("#");
      var _loc10_ = 0;
      var _loc19_ = 0;
      _loc10_;
      var _loc16_ = Number(_loc11_[_loc10_++]);
      _loc10_;
      var _loc18_ = _loc11_[_loc10_++].split("");
      var _loc12_ = _loc18_.length / _loc16_;
      var _loc7_ = new Array(_loc16_);
      var _loc15_ = 0;
      while(_loc15_ < _loc16_)
      {
         _loc7_[_loc15_] = new Array(_loc12_);
         var _loc5_ = 0;
         while(_loc5_ < _loc12_)
         {
            _loc7_[_loc15_][_loc5_] = new Array(0,0,0);
            _loc5_ = _loc5_ + 1;
         }
         _loc15_ = _loc15_ + 1;
      }
      this.spawnPoints = new Array();
      this.crateSpawnPoints = new Array();
      this.grounds = new Array();
      var _loc4_ = 0;
      while(_loc4_ < _loc12_)
      {
         var _loc2_ = 0;
         while(_loc2_ < _loc16_)
         {
            _loc19_;
            var _loc3_ = Number(_loc18_[_loc19_++]);
            var _loc6_ = false;
            var _loc8_ = false;
            var _loc9_ = false;
            if(_loc3_ / 4 >= 1)
            {
               _loc8_ = true;
               _loc3_ %= 4;
            }
            if(_loc3_ / 2 >= 1)
            {
               _loc9_ = true;
               _loc3_ %= 2;
            }
            if(_loc3_ >= 1)
            {
               _loc6_ = true;
            }
            if(_loc6_)
            {
               _loc7_[_loc2_][_loc4_][0] = 1;
            }
            if(_loc9_)
            {
               _loc7_[_loc2_][_loc4_ - 1][1] = 1;
            }
            if(_loc8_)
            {
               _loc7_[_loc2_][_loc4_][2] = 1;
            }
            if(_loc6_)
            {
               this.grounds.push({x:_loc2_,y:_loc4_});
            }
            _loc2_ = _loc2_ + 1;
         }
         _loc4_ = _loc4_ + 1;
      }
      _loc10_;
      var _loc24_ = Number(_loc11_[_loc10_++]);
      _loc10_;
      var _loc23_ = Number(_loc11_[_loc10_++]);
      _loc15_ = 0;
      while(_loc15_ < _loc23_)
      {
         _loc10_;
         var _loc14_ = Number(_loc11_[_loc10_++]);
         _loc10_;
         var _loc13_ = Number(_loc11_[_loc10_++]);
         _loc10_;
         var _loc17_ = Number(_loc11_[_loc10_++]);
         _loc10_;
         var _loc20_ = _loc11_[_loc10_++].split(",");
         switch(_loc17_)
         {
            case 5:
               this.spawnPoints.push({x:_loc14_ - 1,y:_loc13_ - 1});
               break;
            case 8:
               this.crateSpawnPoints.push({x:_loc14_ - 1,y:_loc13_ - 1});
         }
         _loc15_ = _loc15_ + 1;
      }
      _loc10_ = _loc10_ + 1;
      this.playCount = this.playCount + 1;
      this.title = this.data[this.selectedMaze].t;
      this.creator = this.data[this.selectedMaze].n;
      return _loc7_;
   }
   function getSpawnPoints()
   {
      return this.spawnPoints.concat();
   }
   function getCrateSpawnPoints()
   {
      return this.crateSpawnPoints.concat();
   }
   function getGrounds()
   {
      return this.grounds.concat();
   }
   function getPlayCount()
   {
      return this.playCount;
   }
   function getTitle()
   {
      return this.title;
   }
   function getCreator()
   {
      return this.creator;
   }
   function finishWithMaze()
   {
      this.sendRequest(this.selectedMaze);
      this.playCount = 0;
   }
   function sendRequest(num)
   {
      var _loc3_ = undefined;
      if(this.userCount == 0 && this.usernames.length == 0)
      {
         _loc3_ = "c=" + Math.random();
      }
      else
      {
         _loc3_ = "userName=" + this.usernames[Math.floor(Math.random() * this.userCount)];
      }
      _loc3_ = "includes/loadMaze.php?q=" + Base64.Encode(_root.shuffleMessage(_loc3_ + "&a=" + Math.random() + "&b=" + Math.random()));
      this.data[num] = undefined;
      this.loaders[num].loadData(_loc3_,this.data,num);
   }
}
