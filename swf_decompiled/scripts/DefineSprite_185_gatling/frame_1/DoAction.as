onEnterFrame = function()
{
   if(_root.frozen)
   {
      _root.soundGatlingMotorStart.stop("soundGatlingMotorStart");
      _root.soundGatlingMotor.stop("soundGatlingMotor");
      _root.soundGatlingMotorStop.stop("soundGatlingMotorStop");
      return undefined;
   }
   if(!owner.alive)
   {
      _root.soundGatlingMotorStart.stop("soundGatlingMotorStart");
      _root.soundGatlingMotor.stop("soundGatlingMotor");
      _root.soundGatlingMotorStop.stop("soundGatlingMotorStop");
      this.removeMovieClip();
   }
   if(owner.turret._currentframe == 16)
   {
      owner.turret.gotoAndPlay(13);
   }
   if(active && owner.triggerReleased)
   {
      active = false;
      _root.soundGatlingMotorStart.stop("soundGatlingMotorStart");
      _root.soundGatlingMotor.stop("soundGatlingMotor");
      if(_root.soundOn)
      {
         _root.soundGatlingMotorStop.start(2.100000000000001 - 2.100000000000001 * (Math.pow(spinSpeed,0.3) / Math.pow(_root.GATLINGSPINSPEED,0.3)));
      }
      spinSpeed += 35;
   }
   if(active)
   {
      if(spinSpeed < _root.GATLINGSPINSPEED)
      {
         spinSpeed++;
      }
      if(spinSpeed == _root.GATLINGSPINSPEED - 1)
      {
         _root.soundGatlingMotorStart.stop("soundGatlingMotorStart");
         if(_root.soundOn)
         {
            _root.soundGatlingMotor.start(0,999);
         }
      }
      if(spinSpeed == _root.GATLINGSPINSPEED)
      {
         fireCounter++;
         if(fireCounter % 3 == 0)
         {
            if(bulletsLeft > 0)
            {
               bulletsLeft--;
               if(_root.soundOn)
               {
                  _root.soundGatlingShot.start();
               }
               gatlingBulletDepth = _root.game.getNextHighestDepth();
               gatlingBulletName = "gatlingBullet" + gatlingBulletDepth;
               gatlingBullet = _root.game.attachMovie("gatlingBullet",gatlingBulletName,gatlingBulletDepth);
               owner.swapDepths(gatlingBullet);
               gatlingBullet.x = owner._x + Math.cos((owner._rotation - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
               gatlingBullet.y = owner._y + Math.sin((owner._rotation - 90) * 3.141592653589793 / 180) * _root.SCALE * 4.5 / 16;
               gatlingBullet._x = gatlingBullet.x;
               gatlingBullet._y = gatlingBullet.y;
               gatlingBullet._xscale = 100 * (_root.SCALE / 50);
               gatlingBullet._yscale = 100 * (_root.SCALE / 50);
               var _loc3_ = owner._rotation - 90 + Math.random() * 11 - 5.5;
               gatlingBullet.xSpeed = Math.cos(_loc3_ * 3.141592653589793 / 180) * _root.GATLINGSPEED / _root.GATLINGHITCHECKINTERVALS * (_root.SCALE / 50);
               gatlingBullet.ySpeed = Math.sin(_loc3_ * 3.141592653589793 / 180) * _root.GATLINGSPEED / _root.GATLINGHITCHECKINTERVALS * (_root.SCALE / 50);
               gatlingBullet.lifetime = _root.GATLINGLIFETIME;
               gatlingBullet.deadly = _root.GATLINGDEADLY;
               gatlingBullet.owner = owner;
            }
            else if(_root.soundOn)
            {
               _root.soundExposion.start();
            }
         }
      }
   }
   if(!active)
   {
      if(spinSpeed > 0)
      {
         spinSpeed--;
      }
      if(spinSpeed == 0 && owner.gatlingReady)
      {
         this.removeMovieClip();
      }
      if(spinSpeed == 0)
      {
         owner.gatlingReady = true;
         _root.setWeapon(owner,"bullet");
      }
   }
};
